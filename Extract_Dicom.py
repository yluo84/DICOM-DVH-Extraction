import os
import re
import json
import numpy as np
import pydicom
import SimpleITK as sitk
from skimage.draw import polygon

DEBUG = True


def dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# Basic helpers

def label_matching(s, lbl_lst=None):
    if lbl_lst is None:
        lbl_lst = []
    s = re.sub(r'[\s\-_]', '', str(s).lower())
    return s in [re.sub(r'[\s\-_]', '', x.lower()) for x in lbl_lst]


def get_rotation_matrix(patient_orientation):
    direction_x = np.array(patient_orientation[0:3], dtype=np.float64)
    direction_y = np.array(patient_orientation[3:6], dtype=np.float64)
    direction_z = np.cross(direction_x, direction_y)
    rot_mat = np.asarray([direction_x, direction_y, direction_z], dtype=np.float64)
    return rot_mat


def get_sitk_direction_from_iop(iop):
    row_dir = np.array(iop[0:3], dtype=np.float64)
    col_dir = np.array(iop[3:6], dtype=np.float64)
    slice_dir = np.cross(row_dir, col_dir)

    direction = np.eye(3, dtype=np.float64)
    direction[:, 0] = col_dir
    direction[:, 1] = row_dir
    direction[:, 2] = slice_dir
    return direction


def robust_slice_thickness_from_positions(slice_positions, fallback=1.0, eps=1e-4):
    if len(slice_positions) < 2:
        return float(fallback)

    z_vals = np.asarray([p[2] for p in slice_positions], dtype=np.float64)
    diffs = np.abs(np.diff(z_vals))
    valid_diffs = diffs[diffs > eps]

    if len(valid_diffs) > 0:
        return float(np.median(valid_diffs))
    return float(fallback)


def safe_get_frame_uid(ds):
    if hasattr(ds, "FrameOfReferenceUID"):
        return str(ds.FrameOfReferenceUID)
    try:
        return str(ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID)
    except Exception:
        return None


def dcm_modality(ds):
    try:
        return str(ds.Modality)
    except Exception:
        return None


def bbox_3d(mask):
    idx = np.argwhere(mask > 0)
    if len(idx) == 0:
        return None
    return idx.min(axis=0).tolist(), idx.max(axis=0).tolist()


def to_builtin(obj):
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# CT geometry

def load_ct_geometry(ct_img):
    slices = []
    for f in ct_img:
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=False)
            slices.append(ds)
        except Exception:
            continue

    if len(slices) == 0:
        raise ValueError("No CT slices found.")

    iop = slices[0].ImageOrientationPatient
    rot_mat = get_rotation_matrix(iop)
    sitk_direction = get_sitk_direction_from_iop(iop)

    slices.sort(key=lambda x: float(np.matmul(rot_mat, x.ImagePositionPatient)[2]))

    dedup_by_sop = {}
    for s in slices:
        sop_uid = getattr(s, "SOPInstanceUID", None)
        if sop_uid is not None:
            dedup_by_sop[sop_uid] = s
        else:
            dedup_by_sop[id(s)] = s

    slices = list(dedup_by_sop.values())
    slices.sort(key=lambda x: float(np.matmul(rot_mat, x.ImagePositionPatient)[2]))

    unique_slices = []
    seen_z = []
    eps = 1e-3
    for s in slices:
        z = float(np.matmul(rot_mat, s.ImagePositionPatient)[2])
        if not any(abs(z - zz) < eps for zz in seen_z):
            unique_slices.append(s)
            seen_z.append(z)

    slices = unique_slices
    slice_positions = [np.matmul(rot_mat, s.ImagePositionPatient) for s in slices]

    fallback_thickness = float(getattr(slices[0], "SliceThickness", 1.0))
    thickness = robust_slice_thickness_from_positions(
        slice_positions,
        fallback=fallback_thickness,
        eps=1e-4,
    )

    spacing = np.array([
        float(slices[0].PixelSpacing[1]),
        float(slices[0].PixelSpacing[0]),
        thickness,
    ], dtype=np.float64)

    size = np.array([
        int(slices[0].Columns),
        int(slices[0].Rows),
        len(slices),
    ], dtype=int)

    origin = np.array(slices[0].ImagePositionPatient, dtype=np.float64)

    dprint("=== CT geometry debug ===")
    dprint("CT slices after SOP+z dedup:", len(slices))
    dprint("CT spacing:", spacing)
    dprint("CT origin:", origin)

    z_vals = np.asarray([p[2] for p in slice_positions], dtype=np.float64)
    diffs = np.abs(np.diff(z_vals))
    dprint(
        "CT z diff min/max:",
        float(np.min(diffs)) if len(diffs) else None,
        float(np.max(diffs)) if len(diffs) else None,
    )
    dprint("CT nonzero z diff count:", int(np.sum(diffs > 1e-4)))

    info = {
        "origin": origin,
        "spacing": spacing,
        "size": size,
        "rot_mat": rot_mat,
        "sitk_direction": sitk_direction,
    }
    return slices, info

# RTDOSE reading

def read_rtdose_dcm_safe(rd_file):
    ds = pydicom.dcmread(rd_file, stop_before_pixels=False)

    iop = ds.ImageOrientationPatient
    rot_mat = get_rotation_matrix(iop)
    sitk_direction = get_sitk_direction_from_iop(iop)

    origin = np.array(ds.ImagePositionPatient, dtype=np.float64)
    if hasattr(ds, "GridFrameOffsetVector") and len(ds.GridFrameOffsetVector) > 0:
        offsets = np.asarray(ds.GridFrameOffsetVector, dtype=np.float64)
        row_dir = np.array(ds.ImageOrientationPatient[0:3], dtype=np.float64)
        col_dir = np.array(ds.ImageOrientationPatient[3:6], dtype=np.float64)
        slice_dir = np.cross(row_dir, col_dir)
        origin = origin + float(offsets[0]) * slice_dir

    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "DoseGridScaling"):
        arr = arr * float(ds.DoseGridScaling)
    else:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    if hasattr(ds, "GridFrameOffsetVector") and len(ds.GridFrameOffsetVector) > 1:
        offsets = np.asarray(ds.GridFrameOffsetVector, dtype=np.float64)
        diffs = np.abs(np.diff(offsets))
        valid_diffs = diffs[diffs > 1e-4]
        if len(valid_diffs) > 0:
            thickness = float(np.median(valid_diffs))
        else:
            thickness = float(getattr(ds, "SliceThickness", 1.0))
    else:
        thickness = float(getattr(ds, "SliceThickness", 1.0))

    spacing = np.array([
        float(ds.PixelSpacing[1]),
        float(ds.PixelSpacing[0]),
        thickness,
    ], dtype=np.float64)

    size = np.array([arr.shape[2], arr.shape[1], arr.shape[0]], dtype=int)

    info = {
        "origin": origin,
        "spacing": spacing,
        "size": size,
        "rot_mat": rot_mat,
        "sitk_direction": sitk_direction,
        "dose_units": getattr(ds, "DoseUnits", None),
        "dose_type": getattr(ds, "DoseType", None),
        "dose_summation_type": getattr(ds, "DoseSummationType", None),
    }

    dprint("=== RTDOSE debug ===")
    dprint("DoseUnits:", info["dose_units"])
    dprint("DoseType:", info["dose_type"])
    dprint("DoseSummationType:", info["dose_summation_type"])
    dprint("dose array shape:", arr.shape)
    dprint("dose min/max/mean:", float(np.min(arr)), float(np.max(arr)), float(np.mean(arr)))

    return arr.astype(np.float32), info, ds

# Resampling

def resample_to_reference_sitk(images, img_info, reference_info, is_label=False):
    src_img = sitk.GetImageFromArray(images.astype(np.float32))
    src_img.SetOrigin(img_info["origin"].tolist())
    src_img.SetSpacing(img_info["spacing"].tolist())
    src_img.SetDirection(img_info["sitk_direction"].flatten().tolist())

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputOrigin(reference_info["origin"].tolist())
    resampler.SetOutputSpacing(reference_info["spacing"].tolist())
    resampler.SetOutputDirection(reference_info["sitk_direction"].flatten().tolist())
    resampler.SetSize(reference_info["size"].tolist())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0.0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)

    out = resampler.Execute(src_img)
    out_arr = sitk.GetArrayFromImage(out).astype(np.float32)

    dprint("=== Resample debug ===")
    dprint("resampled dose shape:", out_arr.shape)
    dprint("resampled dose min/max/mean:", float(np.min(out_arr)), float(np.max(out_arr)), float(np.mean(out_arr)))
    return out_arr


def read_aligned_dcm_safe(rd_file, ct_img):
    _, ct_info = load_ct_geometry(ct_img)
    dose_arr, dose_info, rd_dcm = read_rtdose_dcm_safe(rd_file)
    dose_map = resample_to_reference_sitk(dose_arr, dose_info, ct_info, is_label=False)
    return dose_map, ct_info["spacing"], ct_info, rd_dcm

# RTSTRUCT -> mask

def read_labels_from_dicomrt_safe(rs_file, ct_img):
    slices, ct_info = load_ct_geometry(ct_img)
    ds = pydicom.dcmread(rs_file, stop_before_pixels=True)

    rot_mat = ct_info["rot_mat"]
    slice_positions = [np.matmul(rot_mat, s.ImagePositionPatient) for s in slices]
    z = np.asarray([s[2] for s in slice_positions], dtype=np.float32)

    spacing_r = float(slices[0].PixelSpacing[0])
    spacing_c = float(slices[0].PixelSpacing[1])
    shape = (len(slices), slices[0].Rows, slices[0].Columns)

    rois = []
    contours = []
    roi_id = 0

    if not hasattr(ds, "ROIContourSequence"):
        return [], np.zeros((0,) + shape, dtype=bool), ct_info

    for i in range(len(ds.ROIContourSequence)):
        try:
            if not hasattr(ds.ROIContourSequence[i], "ContourSequence"):
                continue

            contour = {}
            roi_contours = []
            for s in ds.ROIContourSequence[i].ContourSequence:
                if getattr(s, "ContourGeometricType", "") != "CLOSED_PLANAR":
                    continue
                con = s.ContourData
                if isinstance(con, bytes):
                    con = list(map(float, con.decode("utf-8").split("\\")))
                roi_contours.append(con)

            if len(roi_contours) == 0:
                continue

            contour["contours"] = roi_contours
            roi_id += 1

            roi = {
                "id": str(roi_id),
                "number": ds.StructureSetROISequence[i].ROINumber,
                "name": ds.StructureSetROISequence[i].ROIName,
                "color": getattr(ds.ROIContourSequence[i], "ROIDisplayColor", [255, 255, 255]),
            }
            contour["id"] = roi_id
            rois.append(roi)
            contours.append(contour)
        except Exception:
            continue

    label_map = np.zeros((len(rois),) + shape, dtype=bool)

    for contour in contours:
        is_label = np.zeros(shape, dtype=np.float32)
        num = contour["id"] - 1

        for con in contour["contours"]:
            nodes = np.array(con, dtype=np.float32).reshape((-1, 3))
            if nodes.shape[0] < 3:
                continue

            transformed_nodes = np.matmul(nodes, np.transpose(rot_mat))
            if np.amax(np.abs(np.diff(transformed_nodes[:, 2]))) > 0.05:
                continue

            z_index = np.argmin(np.abs(z - transformed_nodes[0, 2]))
            pos_r = slice_positions[z_index][1]
            pos_c = slice_positions[z_index][0]

            r = (transformed_nodes[:, 1] - pos_r) / spacing_r
            c = (transformed_nodes[:, 0] - pos_c) / spacing_c
            r = np.clip(r, 0.5, shape[1] - 0.5)
            c = np.clip(c, 0.5, shape[2] - 0.5)

            rr, cc = polygon(r, c, shape=(shape[1], shape[2]))
            is_label[z_index, rr, cc] = 1 - is_label[z_index, rr, cc]

        label_map[num][is_label > 0] = 1

    return rois, label_map, ct_info


# Dose metrics

def get_roi_voxel_dose(dose_map, roi_mask):
    vals = dose_map[roi_mask > 0]
    vals = vals[np.isfinite(vals)]
    return vals.astype(np.float32)


def compute_d90(dose_values):
    if len(dose_values) == 0:
        return None
    return float(np.percentile(dose_values, 10))


def compute_d2cc(dose_values, voxel_spacing):
    if len(dose_values) == 0:
        return None
    voxel_volume_cc = float(np.prod(voxel_spacing) / 1000.0)
    n_vox = int(np.ceil(2.0 / voxel_volume_cc))
    n_vox = max(1, min(n_vox, len(dose_values)))
    vals = np.sort(dose_values)[::-1]
    return float(vals[n_vox - 1])


def build_dvh(dose_values, voxel_spacing, bin_width=0.1):
    if len(dose_values) == 0:
        return [], {"min_dose": None, "max_dose": None, "mean_dose": None, "d90": None}

    voxel_volume_cc = float(np.prod(voxel_spacing) / 1000.0)
    max_dose = float(np.max(dose_values))
    bins = np.arange(0, max_dose + bin_width, bin_width)
    if len(bins) < 2:
        bins = np.array([0.0, max_dose + bin_width], dtype=np.float32)

    hist, edges = np.histogram(dose_values, bins=bins)
    cumulative_volume = np.cumsum(hist[::-1])[::-1] * voxel_volume_cc
    dose_axis = edges[:-1]

    dvh_data = np.column_stack((dose_axis, cumulative_volume)).tolist()
    dvh_info = {
        "min_dose": float(np.min(dose_values)),
        "max_dose": float(np.max(dose_values)),
        "mean_dose": float(np.mean(dose_values)),
        "d90": float(np.percentile(dose_values, 10)),
    }
    return dvh_data, dvh_info


# RTPLAN helpers

def extract_dwell_points_info(rp_dcm):
    dwell_points_info = {}
    if not hasattr(rp_dcm, "ApplicationSetupSequence"):
        return dwell_points_info

    try:
        app_setup = rp_dcm.ApplicationSetupSequence[0]
        if not hasattr(app_setup, "ChannelSequence"):
            return dwell_points_info

        for cha in app_setup.ChannelSequence:
            source_applicator_name = str(getattr(cha, "SourceApplicatorID", "Unknown"))
            total_t = float(getattr(cha, "ChannelTotalTime", 0.0))
            final_cumulative = float(getattr(cha, "FinalCumulativeTimeWeight", 1.0))
            pts_seq = cha.BrachyControlPointSequence

            source_pts = [list(map(float, pts.ControlPoint3DPosition)) for pts in pts_seq[::2]]
            dwell_time = []
            for i in range(0, len(pts_seq), 2):
                dt = float(pts_seq[i + 1].CumulativeTimeWeight) - float(pts_seq[i].CumulativeTimeWeight)
                dt = dt / final_cumulative * total_t if final_cumulative != 0 else 0.0
                dwell_time.append(float(dt))

            dwell_points_info[source_applicator_name] = {
                "dwell_positions": source_pts,
                "dwell_time": dwell_time,
            }
    except Exception as e:
        dprint("extract_dwell_points_info warning:", e)

    return dwell_points_info


def get_prescribed_dose_safe(rp_dcm):
    try:
        fgs = rp_dcm.FractionGroupSequence
        if len(fgs) > 0 and hasattr(fgs[0], "BrachyApplicationSetupDose"):
            val = float(fgs[0].BrachyApplicationSetupDose)
            if val > 0:
                return val
    except Exception:
        pass

    try:
        drs = rp_dcm.DoseReferenceSequence
        for dr in drs:
            if hasattr(dr, "TargetPrescriptionDose"):
                val = float(dr.TargetPrescriptionDose)
                if val > 0:
                    return val
    except Exception:
        pass

    try:
        drs = rp_dcm.DoseReferenceSequence
        for dr in drs:
            if hasattr(dr, "DeliveryMaximumDose"):
                val = float(dr.DeliveryMaximumDose)
                if val > 0:
                    return val
    except Exception:
        pass

    return None

# Core extraction

def get_plan_data_safe(rs_file, rp_file, rd_file, ct_img):
    dprint(">>> RUNNING FIXED VERSION <<<")

    rois, label_map, ct_info = read_labels_from_dicomrt_safe(rs_file, ct_img)
    dose_map, voxel_spacing, _, rd_dcm = read_aligned_dcm_safe(rd_file, ct_img)
    rp_dcm = pydicom.dcmread(rp_file, stop_before_pixels=True)

    dprint("=== BBOX debug ===")
    dprint("dose nonzero bbox:", bbox_3d(dose_map > 0))

    roi_aliases = [
        ["ctv", "hrctv"],
        ["bladder"],
        ["rectum"],
        ["sigmoid"],
        ["smallbowel", "small_bowel", "bowel", "small bowel"],
    ]

    dvh = {
        "dose_scaling": 1,
        "volume_units": "CM3",
        "dvh_type": "CUMULATIVE",
        "dose_units": getattr(rd_dcm, "DoseUnits", "GY"),
        "dose_type": getattr(rd_dcm, "DoseType", ""),
        "dose_plot_cut_off": 40,
        "roi_dvh": [],
    }

    cumulative_data = {
        "plan_name": getattr(rp_dcm, "SeriesDescription", ""),
        "plan_date": getattr(rp_dcm, "SeriesDate", ""),
        "plan_time": getattr(rp_dcm, "SeriesTime", ""),
        "prescribed_dose": get_prescribed_dose_safe(rp_dcm),
    }

    d2cc_lst = {}
    roi_stats = {}

    for roi_group in roi_aliases:
        matched = [r for r in rois if label_matching(r["name"], roi_group)]
        if not matched:
            continue

        roi = matched[0]
        mask = label_map[int(roi["id"]) - 1]
        dose_values = get_roi_voxel_dose(dose_map, mask)

        dprint(f"=== ROI debug: {roi['name']} ===")
        dprint("roi bbox:", bbox_3d(mask))
        dprint("voxel count:", int(len(dose_values)))
        if len(dose_values) > 0:
            dprint(
                "dose min/max/mean:",
                float(np.min(dose_values)),
                float(np.max(dose_values)),
                float(np.mean(dose_values)),
            )

        if len(dose_values) == 0:
            continue

        roi_stats[roi["name"]] = {
            "voxel_count": int(len(dose_values)),
            "min_dose": float(np.min(dose_values)),
            "max_dose": float(np.max(dose_values)),
            "mean_dose": float(np.mean(dose_values)),
            "d90": float(compute_d90(dose_values)),
        }

        dvh_data, dvh_info = build_dvh(dose_values, voxel_spacing)
        dvh["roi_dvh"].append({
            "roi_name": roi["name"],
            "reference_roi_id": roi["id"],
            "dvh_data": dvh_data,
            "dvh_info": dvh_info,
            "color": ", ".join([str(i) for i in roi["color"]]),
        })

        normalized_group = [x.replace(" ", "").lower() for x in roi_group]
        if "ctv" in normalized_group:
            cumulative_data["hrctv_d90"] = round(compute_d90(dose_values), 2)
        else:
            d2cc = round(compute_d2cc(dose_values, voxel_spacing), 2)
            d2cc_lst[roi_group[0]] = d2cc
            cumulative_data[f"{roi_group[0]}_d2cc"] = d2cc
            roi_stats[roi["name"]]["d2cc"] = float(d2cc)

    dwell_points_info = extract_dwell_points_info(rp_dcm)

    clinical_summary = {
        "target": {
            "hrctv_d90": cumulative_data.get("hrctv_d90"),
            "prescribed_dose": cumulative_data.get("prescribed_dose"),
        },
        "oars": {
            "bladder_d2cc": cumulative_data.get("bladder_d2cc"),
            "rectum_d2cc": cumulative_data.get("rectum_d2cc"),
            "sigmoid_d2cc": cumulative_data.get("sigmoid_d2cc"),
            "smallbowel_d2cc": cumulative_data.get("smallbowel_d2cc"),
        },
        "implant": {
            "num_channels": len(dwell_points_info),
            "channel_ids": list(dwell_points_info.keys()),
        },
    }

    config = {
        "dwell_points_info": dwell_points_info,
        "input_config": {},
        "dvh": dvh,
        "d2cc": d2cc_lst,
        "roi_stats": roi_stats,
        "cumulative_data": cumulative_data,
        "clinical_summary": clinical_summary,
    }

    dprint("=== FINAL cumulative_data ===")
    dprint(json.dumps(cumulative_data, indent=2, default=str))
    return config

# Auto scan cases from root

def auto_find_complete_case(search_root):
    rs_files = []
    rp_files = []
    rd_files = []
    ct_groups = {}

    dprint(f"--- scanning {search_root} ---")

    for root, _, files in os.walk(search_root):
        for f in files:
            file_path = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                modality = dcm_modality(ds)
                frame_uid = safe_get_frame_uid(ds)

                if modality == "RTSTRUCT":
                    rs_files.append((file_path, frame_uid))
                elif modality == "RTPLAN":
                    rp_files.append((file_path, frame_uid))
                elif modality == "RTDOSE":
                    rd_files.append((file_path, frame_uid))
                elif modality == "CT":
                    series_uid = getattr(ds, "SeriesInstanceUID", None)
                    sop_uid = getattr(ds, "SOPInstanceUID", None)
                    key = (frame_uid, series_uid)

                    if key not in ct_groups:
                        ct_groups[key] = {}
                    if sop_uid is not None:
                        ct_groups[key][sop_uid] = file_path
                    else:
                        ct_groups[key][file_path] = file_path
            except Exception:
                continue

    dprint("\n--- summary ---")
    dprint("RS:", len(rs_files))
    dprint("RP:", len(rp_files))
    dprint("RD:", len(rd_files))
    dprint("CT series groups:", len(ct_groups))

    for rs_file, rs_uid in rs_files:
        if rs_uid is None:
            continue

        matched_rp = [x for x in rp_files if x[1] == rs_uid]
        matched_rd = [x for x in rd_files if x[1] == rs_uid]
        matched_ct_series = [
            list(file_dict.values())
            for (frame_uid, _series_uid), file_dict in ct_groups.items()
            if frame_uid == rs_uid
        ]

        if len(matched_rp) > 0 and len(matched_rd) > 0 and len(matched_ct_series) > 0:
            ct_img = max(matched_ct_series, key=len)
            
            # Pick the MOST RECENT RP and RD by file modification time
            rp_file = max(matched_rp, key=lambda x: os.path.getmtime(x[0]))[0]
            rd_file = max(matched_rd, key=lambda x: os.path.getmtime(x[0]))[0]

            print("\n Matched complete case")
            print("RS:", rs_file)
            print(f"RP: {rp_file}  (newest of {len(matched_rp)})")
            print(f"RD: {rd_file}  (newest of {len(matched_rd)})")
            print("CT count:", len(ct_img))
            print("Frame UID:", rs_uid)
            return rs_file, rp_file, rd_file, ct_img

    print("\n No complete case found.")
    return None, None, None, None

# Main

if __name__ == "__main__":
    search_root = r"C:\carina"

    rs_file, rp_file, rd_file, ct_img = auto_find_complete_case(search_root)

    if rs_file and rp_file and rd_file and ct_img:
        config = get_plan_data_safe(rs_file, rp_file, rd_file, ct_img)

        save_path = os.path.join(os.path.dirname(rp_file), "plan_metrics.json")
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(to_builtin(config), f, indent=2, ensure_ascii=False)

        print(f"\nSaved JSON to: {save_path}")

        print("\n==== cumulative_data ====")
        print(config["cumulative_data"])

        print("\n==== clinical_summary ====")
        print(config["clinical_summary"])

        print("\n==== dwell_points_info ====")
        for k, v in config["dwell_points_info"].items():
            print(k, v)
    else:
        print("No complete case found.")
