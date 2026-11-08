#!/usr/bin/env python3
"""Run unchanged SHEETS q-1.0 on shared preparation and shared JHU regions.

The estimator, sampler, aggregation, regional statistics and QA renderer are
imported unchanged. This adapter supplies already prepared images, disables the
engine's own registration, and records the supplied shared regions explicitly.
The coordinator owns final cleanup; this stage always retains engine work.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True

import nibabel as nib
import numpy as np

DEFAULT_RADIUS_MM = 6.875
DEFAULT_MEASUREMENT = dict(draws=100, radius_mm=DEFAULT_RADIUS_MM, qref=.03,
                           seed=20261015, sample_batch_size=4)


def validate_radius(radius_mm, spacing_mm=None):
    """Match the frozen estimator's positive radius and int16 cube limits."""
    try:
        radius_mm = float(radius_mm)
    except (TypeError, ValueError) as error:
        raise ValueError("--radius-mm must be finite and positive") from error
    if not math.isfinite(radius_mm) or radius_mm <= 0:
        raise ValueError("--radius-mm must be finite and positive")
    if spacing_mm is not None:
        # The frozen isotropic evaluator uses h=ceil(radius/spacing)-1 and
        # (2h+1)^3 <=32767. Its largest supported cube is 31^3: radius <=16s.
        ratio = radius_mm / float(spacing_mm)
        if not math.isfinite(ratio) or ratio > 16:
            raise ValueError("--radius-mm exceeds the frozen int16 neighborhood limit "
                             f"({16*float(spacing_mm):g} mm on this grid)")
    return radius_mm


def measurement_settings(args):
    """Normalize the original CLI types and bounds before reading large inputs."""
    settings = dict(radius_mm=validate_radius(args.radius_mm))
    for key, minimum in (("draws", 1), ("seed", 0), ("sample_batch_size", 1)):
        try:
            value = operator.index(getattr(args, key))
        except TypeError as error:
            raise ValueError(f"--{key.replace('_', '-')} must be an integer >= {minimum}") from error
        if isinstance(getattr(args, key), (bool, np.bool_)) or value < minimum:
            raise ValueError(f"--{key.replace('_', '-')} must be an integer >= {minimum}")
        settings[key] = value
    try:
        settings["qref"] = float(args.qref)
    except (TypeError, ValueError) as error:
        raise ValueError("--qref must be finite and positive") from error
    if not math.isfinite(settings["qref"]) or settings["qref"] <= 0:
        raise ValueError("--qref must be finite and positive")
    for key, value in settings.items():
        setattr(args, key, value)
    return settings


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def engine():
    root = Path(os.environ.get("SHEETS_ROOT", "/opt/sheets-q"))
    if not root.is_dir() and "SHEETS_ROOT" not in os.environ:
        root = Path(__file__).resolve().parent.parent / "Software"
    sys.path[:0] = [str(root / "Measurement"), str(root / "Code/spi_python")]
    import sheets_single
    if sheets_single.VERSION != "q-1.0":
        raise RuntimeError("This adapter requires the frozen SHEETS q-1.0 engine")
    if Path(sheets_single.__file__).resolve().parent != (root / "Measurement").resolve():
        raise RuntimeError("A different SHEETS engine is already imported")
    return sheets_single


def label_names(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = {}
    for row in rows:
        label = int(row["id"])
        if label <= 0 or label in names or not row["name"].strip():
            raise ValueError("Label table must contain unique positive IDs and names")
        names[label] = row["name"].strip()
    if set(names) != set(range(1, 51)):
        raise ValueError("Shared JHU table must contain all 50 IDs (1 through 50)")
    return names


def validate_shared(args):
    """Enforce one existing isotropic grid; never resample a shared image."""
    reference = nib.load(args.fod)
    if reference.ndim != 4 or reference.shape[-1] not in (28, 45):
        raise ValueError("Shared FOD must be Tournier07 lmax 6 or 8")
    affine = np.asarray(reference.affine, float)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    if (not np.isfinite(affine).all() or np.any(spacing <= 0)
            or not np.allclose(affine[3], [0, 0, 0, 1])
            or not np.allclose(spacing, spacing[0], atol=1e-6, rtol=1e-6)
            or not np.allclose((affine[:3, :3] / spacing).T @
                               (affine[:3, :3] / spacing), np.eye(3), atol=1e-6, rtol=0)):
        raise ValueError("Shared FOD requires an orthogonal isotropic grid")
    validate_radius(getattr(args, "radius_mm", DEFAULT_RADIUS_MM), spacing[0])
    names = label_names(args.labels_table) if args.labels_table else None
    if bool(args.labels) != bool(args.labels_table):
        raise ValueError("--labels and --labels-table must be supplied together")
    if (args.eroded_labels or args.template_fa) and not args.labels:
        raise ValueError("Shared erosion/template inputs require --labels")
    values = {}
    for name in ("brain_mask", "wm_mask", "fa", "b0", "labels", "eroded_labels", "template_fa"):
        path = getattr(args, name, None)
        if path is None:
            continue
        image = nib.load(path)
        if image.shape != reference.shape[:3] or not np.allclose(image.affine, affine, atol=1e-5, rtol=0):
            raise ValueError(f"Shared {name} must exactly match the FOD grid")
        data = np.asarray(image.dataobj)
        if name in ("brain_mask", "wm_mask"):
            if not np.isfinite(data).all() or not np.any(data > 0):
                raise ValueError(f"Shared {name} is empty or nonfinite")
            values[name] = data > 0
        if name in ("labels", "eroded_labels"):
            if (not np.isfinite(data).all() or not np.array_equal(data, np.rint(data))
                    or not set(np.unique(data)).issubset({0, *names})):
                raise ValueError(f"Shared {name} must contain only integer JHU IDs 0 through 50")
            values[name] = data
    if not np.any(values["brain_mask"] & values["wm_mask"]):
        raise ValueError("Shared WM and brain masks have no overlap")
    if "eroded_labels" in values:
        eroded = values["eroded_labels"]
        if np.any((eroded != 0) & (eroded != values["labels"])):
            raise ValueError("Eroded labels must be a same-label subset of native labels")
    return dict(shape=list(reference.shape[:3]), affine=affine.tolist(),
                spacing_mm=spacing.tolist(), resampled=False), names


def report_view(source, target, labels=None):
    """Make a disposable engine-layout view; all image links are read-only uses."""
    source, target = Path(source).resolve(), Path(target)
    for name in ("q.nii.gz", "wm_mask.nii.gz", "diagnostics", "tensors", "display",
                 "legacy_SPI", "anatomy", "summary.csv"):
        path = source / name
        if path.exists():
            (target / name).symlink_to(path, target_is_directory=path.is_dir())
    if labels is None:
        if (source / "registration").exists():
            (target / "registration").symlink_to(source / "registration", target_is_directory=True)
    else:
        (target / "registration").mkdir()
        (target / "registration/JHU_labels.nii.gz").symlink_to(Path(labels).resolve())
        (target / "registration/JHU_labels.csv").symlink_to(source / "registration/JHU_labels.csv")


@contextmanager
def published_qa_paths(enabled):
    """Change only frozen report guide text to the published subject layout."""
    if not enabled:
        yield
        return
    from sheets_qa import _Renderer
    original = _Renderer.finish
    changes = []

    def finish(renderer, fig, pdf, title, note=""):
        if title == "What ran":
            regional_guide = ("Shared atlas labels: shared/regions/"
                              if (renderer.out / "registration/JHU_labels.nii.gz").is_file()
                              else "JHU regions: not requested")
            replacements = {
                "Files: q + wm_mask | diagnostics/ | display/ | tensors/ | legacy_SPI/ | anatomy/ | registration/":
                    "Files: sheets/{q.nii.gz, wm_mask.nii.gz, diagnostics/, tensors/, display/, legacy_SPI/}\n"
                    "Shared anatomy: shared/anatomy/ | " + regional_guide,
                "Full provenance is in run.json; per-pair whole-WM and regional statistics are in summary.csv.":
                    "Run details: run.json | Statistics: sheets/summary.csv | Engine records: provenance/modules/sheets/",
            }
            for before, after in replacements.items():
                matches = [item for item in fig.texts if item.get_text() == before]
                if len(matches) != 1:
                    raise RuntimeError("Frozen Sheets report guide changed; expected exactly one: " + before)
                matches[0].set_text(after)
                matches[0].set_fontsize(8)
                matches[0].set_linespacing(1.2)
                changes.append(before)
        if title == "JHU regions":
            before = "All pairs and full names: summary.csv."
            if note.count(before) != 1:
                raise RuntimeError("Frozen Sheets regional report guide changed")
            note = note.replace(before, "All pairs and full names: sheets/summary.csv.")
        return original(renderer, fig, pdf, title, note)

    _Renderer.finish = finish
    try:
        yield
        if len(changes) != 2:
            raise RuntimeError("Published Sheets QA did not update both recipe guide lines")
    finally:
        _Renderer.finish = original


def regenerate_qa(module_root, out_pdf, *, replace_existing=False, published_paths=False):
    """Render compact saved results to a new PDF without altering source files."""
    engine()
    from sheets_qa import create_qa
    module_root, out_pdf = Path(module_root).resolve(), Path(out_pdf).resolve()
    if out_pdf == Path("/nfs2") or Path("/nfs2") in out_pdf.parents:
        raise ValueError("Writing under /nfs2 is prohibited")
    if out_pdf.exists() and not replace_existing:
        raise FileExistsError(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((module_root / "run.json").read_text())
    with tempfile.TemporaryDirectory(prefix="sheets-qa-", dir=out_pdf.parent) as directory:
        temporary = Path(directory)
        report_view(module_root, temporary)
        with published_qa_paths(published_paths):
            record = create_qa(temporary, metadata)
        (temporary / "qa.pdf").replace(out_pdf)
    record["path"] = str(out_pdf)
    record["shared_registration"] = metadata.get("shared_registration")
    record["adapter_sha256"] = sha(__file__)
    record["guide_paths"] = "published subject layout" if published_paths else "original module layout"
    return record


def attach_shared_regions(out, args, names, metadata):
    """Use the original regional math and QA on the shared 50-label atlas."""
    if args.labels is None:
        return
    implementation = engine()
    registration = out / "registration"
    registration.mkdir(exist_ok=True)
    shutil.copy2(args.labels, registration / "JHU_labels.nii.gz")
    with (registration / "JHU_labels.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "name"])
        writer.writerows(sorted(names.items()))
    if args.template_fa:
        shutil.copy2(args.template_fa, registration / "template_in_native_FA.nii.gz")
        method = "Shared ANTs FMRIB58-to-diffusion registration supplied by coordinator"
        atlas = "JHU-ICBM labels / FMRIB58 FA"
        status = "Shared ANTs / FMRIB58 FA; JHU 50-label atlas; native labels; no Sheets registration rerun"
    else:
        method = "Provided shared JHU labels already aligned to diffusion; no registration run"
        atlas = "Provided JHU-ICBM 50-label atlas"
        status = "Provided aligned JHU 50-region labels; no registration run"
    metadata["labels_file"] = "registration/JHU_labels.nii.gz"
    metadata["shared_registration"] = dict(
        method=method, atlas=atlas, named_regions=50,
        native_labels="registration/JHU_labels.nii.gz",
        eroded_labels="registration/JHU_labels_eroded.nii.gz" if args.eroded_labels else None,
        region_denominator="Shared label intersect effective Sheets WM; all WM centers remain denominators",
        registration_rerun=False, original_sheets_registration="Disabled (--no-jhu)")
    metadata["jhu_registration"] = dict(status=status)
    implementation.write_summary(out, metadata)
    metadata["regional_summaries"] = {"summary.csv": "native shared JHU labels; original Sheets regional statistics"}
    if args.eroded_labels:
        from verify_single_scan import verify
        eroded = registration / "JHU_labels_eroded.nii.gz"
        shutil.copy2(args.eroded_labels, eroded)
        with tempfile.TemporaryDirectory(prefix="eroded-summary-", dir=out / "work") as directory:
            temporary = Path(directory)
            report_view(out, temporary, labels=eroded)
            # write_summary must write into the disposable view, not through a link.
            (temporary / "summary.csv").unlink()
            implementation.write_summary(temporary, metadata)
            check_metadata = dict(metadata, outputs={})
            write_json(temporary / "run.json", check_metadata)
            check = verify(temporary, keepdraws=False)
            if check["status"] != "passed":
                raise RuntimeError("Shared eroded summary failed original Sheets verifier: " + str(check["errors"]))
            shutil.copy2(temporary / "summary.csv", out / "summary_eroded.csv")
        metadata["regional_summaries"]["summary_eroded.csv"] = "supplied eroded shared JHU labels; original Sheets regional statistics"


def run(args):
    # Normalize the small Namespace API so programmatic callers need not fill options.
    for key, default in dict(b0=None, labels=None, eroded_labels=None, template_fa=None,
                             labels_table=None, preparation_json=None, subject=None,
                             save_draws=False, resume=False, threads=8,
                             **DEFAULT_MEASUREMENT).items():
        if not hasattr(args, key):
            setattr(args, key, default)
    settings = measurement_settings(args)
    out = Path(args.out).expanduser().resolve()
    if out == Path("/nfs2") or Path("/nfs2") in out.parents:
        raise ValueError("Writing under /nfs2 is prohibited")
    paths = {}
    for name in ("fod", "brain_mask", "wm_mask", "fa", "b0", "labels", "eroded_labels",
                 "template_fa", "labels_table", "preparation_json"):
        value = getattr(args, name, None)
        if value is not None:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            if path == out or out in path.parents:
                raise ValueError("Shared inputs must be outside the Sheets module output")
            setattr(args, name, path)
            paths[name] = dict(path=str(path))
    grid, names = validate_shared(args)
    for record in paths.values():
        record["sha256"] = sha(record["path"])
    signature_data = dict(inputs=paths, threads=int(args.threads), subject=args.subject,
                          save_draws=bool(args.save_draws), adapter_sha256=sha(__file__), **settings)
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    state_path = out / "adapter.json"
    old = json.loads(state_path.read_text()) if state_path.exists() else None
    if out.exists() and any(out.iterdir()):
        if not args.resume or old is None:
            raise FileExistsError("Use a new module output, or --resume for this adapter's interrupted run")
        if old["signature"] != signature:
            raise ValueError("Sheets adapter resume inputs or settings differ")
        if old["status"] == "complete":
            engine()  # Make the frozen verifier available in a fresh resume process.
            from verify_single_scan import verify
            result = verify(out)
            if result["status"] != "passed":
                raise RuntimeError("Completed Sheets outputs failed verification: " + str(result["errors"]))
            return json.loads((out / "run.json").read_text())
    implementation = engine()
    out.mkdir(parents=True, exist_ok=True)
    state = dict(status="running", signature=signature, **signature_data)
    write_json(state_path, state)
    try:
        current = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else None
        if current is None or current.get("status") != "complete":
            argv = ["--fod", str(args.fod), "--mask", str(args.brain_mask),
                    "--wm-mask", str(args.wm_mask), "--fa", str(args.fa),
                    "--out", str(out), "--threads", str(args.threads), "--native-spacing",
                    "--draws", str(args.draws), "--radius-mm", str(args.radius_mm),
                    "--seed", str(args.seed), "--qref", str(args.qref),
                    "--sample-batch-size", str(args.sample_batch_size),
                    "--no-jhu", "--keep-work", "--resume"]
            if args.b0:
                argv += ["--b0", str(args.b0)]
            if args.subject:
                argv += ["--subject", str(args.subject)]
            if args.save_draws:
                argv.append("--keep-draws")
            # Internal --resume permits the adapter's prewritten ownership record.
            # The adapter above rejects unrelated/existing completed output folders.
            implementation.main(argv)
        provenance = out / "provenance"
        provenance.mkdir(exist_ok=True)
        original_manifest = provenance / "sheets_engine_run.json"
        if not original_manifest.exists():
            shutil.copy2(out / "run.json", original_manifest)
            shutil.copy2(out / "verification.json", provenance / "sheets_engine_verification.json")
        metadata = json.loads(original_manifest.read_text())
        # Original scientific outputs must survive any failed/retried reporting step.
        for name, record in metadata["outputs"].items():
            if name not in ("summary.csv", "qa.pdf", "verification.json", "adapter.json"):
                if sha(out / name) != record["sha256"]:
                    raise RuntimeError("Original Sheets output changed: " + name)
        metadata["outputs"] = {}
        metadata["adapter"] = dict(version="1", sha256=sha(__file__),
            original_engine_manifest="provenance/sheets_engine_run.json", shared_grid=grid,
            shared_inputs=paths, measurement_settings=settings,
            default_measurement_settings=DEFAULT_MEASUREMENT,
            measurement_settings_changed=settings != DEFAULT_MEASUREMENT,
            note="Shared reconstruction/ANTs regions replace standalone preparation; q estimator is unchanged")
        if args.labels and not args.template_fa:
            metadata["adapter"]["note"] = (
                "Shared reconstruction and provided aligned JHU labels replace standalone preparation; "
                "no atlas registration run; q estimator is unchanged")
        if args.preparation_json:
            preparation = json.loads(args.preparation_json.read_text())
            shutil.copy2(args.preparation_json, provenance / "shared_preparation.json")
            metadata["reconstruction_profile"] = preparation.get("reconstruction", preparation.get("profile", preparation))
            metadata["shared_preparation_file"] = "provenance/shared_preparation.json"
        attach_shared_regions(out, args, names, metadata)
        write_json(out / "run.json", metadata)
        metadata["qa"] = regenerate_qa(out, out / "qa.pdf", replace_existing=True)
        from verify_single_scan import verify
        check = verify(out)
        if check["status"] != "passed":
            raise RuntimeError("Shared Sheets outputs failed original verifier: " + str(check["errors"]))
        write_json(out / "verification.json", check)
        metadata["verification"] = check
        for record in paths.values():
            if sha(record["path"]) != record["sha256"]:
                raise RuntimeError("Shared input changed during Sheets execution: " + record["path"])
        state["status"] = "complete"
        write_json(state_path, state)
        metadata["outputs"] = {str(path.relative_to(out)): dict(bytes=path.stat().st_size, sha256=sha(path))
            for path in sorted(out.rglob("*")) if path.is_file()
            and "work" not in path.relative_to(out).parts and path != out / "run.json"}
        metadata["output_bytes"] = sum(record["bytes"] for record in metadata["outputs"].values())
        write_json(out / "run.json", metadata)
        return metadata
    except Exception as error:
        state.update(status="failed", error=str(error))
        write_json(state_path, state)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("fod", "brain-mask", "wm-mask", "fa", "out"):
        result.add_argument("--" + name, type=Path, required=True)
    for name in ("b0", "labels", "eroded-labels", "template-fa", "labels-table", "preparation-json"):
        result.add_argument("--" + name, type=Path)
    result.add_argument("--threads", type=int, default=8)
    result.add_argument("--radius-mm", type=float, default=DEFAULT_RADIUS_MM,
                        help="Sheets physical fitting radius in mm (default 6.875; finite positive, at most 16 times grid spacing)")
    result.add_argument("--draws", type=int, default=DEFAULT_MEASUREMENT["draws"],
                        help="Positive number of structural FOD draws (default 100)")
    result.add_argument("--seed", type=int, default=DEFAULT_MEASUREMENT["seed"],
                        help="Nonnegative sampling seed (default 20261015)")
    result.add_argument("--qref", type=float, default=DEFAULT_MEASUREMENT["qref"],
                        help="Positive display/tensor reference in inverse mm (default 0.03)")
    result.add_argument("--sample-batch-size", type=int, default=DEFAULT_MEASUREMENT["sample_batch_size"],
                        help="Positive number of draws per engine batch (default 4)")
    result.add_argument("--subject")
    result.add_argument("--save-draws", action="store_true")
    result.add_argument("--resume", action="store_true")
    return result


if __name__ == "__main__":
    if "--qa-from" in sys.argv[1:]:
        qa_parser = argparse.ArgumentParser(description="Regenerate Sheets QA from compact saved output")
        qa_parser.add_argument("--qa-from", type=Path, required=True)
        qa_parser.add_argument("--qa-out", type=Path, required=True)
        qa_parser.add_argument("--published-paths", action="store_true")
        qa_args = qa_parser.parse_args()
        record = regenerate_qa(qa_args.qa_from, qa_args.qa_out, published_paths=qa_args.published_paths)
        print(json.dumps(dict(path=record["path"], page_count=record["page_count"],
                              sha256=sha(qa_args.qa_out))))
    else:
        run(parser().parse_args())
