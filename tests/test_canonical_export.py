"""Canonical-export adapter tests: mapper, round-trip, and bundle pathways.

These are this repo's first conformance-style tests against the v0.2 wire
contract (``canonical_snapshot``): the mapper unit tests pin the native →
nested translation (including the provenance-status vocabulary mapping and
the unknown-state → hidden-marker path), and the bundle tests exercise all
three export pathways end to end on the repo fixtures.
"""

import json
import os

import pytest

# Optional for core local tests; the canonical-contract CI job installs the
# pinned contract and rejects skipped conformance tests.
pytest.importorskip("canonical_snapshot")

from canonical_snapshot import validate_snapshot  # noqa: E402
from cubase_session_explorer.canonical_export import exporter  # noqa: E402
from cubase_session_explorer.canonical_export.mapper import (
    session_state_to_canonical,
    to_canonical,
    to_native,
)
from cubase_session_explorer.models import (
    DeviceState,
    FolderState,
    ProjectMeta,
    SessionState,
    TrackState,
    UnknownState,
)
from cubase_session_explorer.provenance import (
    annotation,
    conflicting_note,
    exported,
    inferred,
    unavailable,
)

BUNDLE_FILES = set(exporter.BUNDLE_FILES)


# ---------------------------------------------------------------------------
# mapper unit tests
# ---------------------------------------------------------------------------


def _make_state() -> SessionState:
    """A hand-built native SessionState covering the mapping edge cases."""
    kick = TrackState(id="cubase:track-kick", name="Kick", track_type="audio",
                      volume_db=-3.0, parent_id="cubase:folder-drums")
    kick.provenance = exported(source_type="dawproject")
    kick.field_provenance["volume_db"] = exported(source_type="dawproject")
    kick.field_provenance["role"] = inferred("role guessed from name", confidence=0.55)
    kick.field_provenance["pan"] = conflicting_note(
        "structural source said 0.0, runtime observed 0.2"
    )
    kick.role = "Drums"
    kick.devices.append(
        DeviceState(
            id="cubase:device-eq", track_id="cubase:track-kick", name="StudioEQ",
            field_provenance={
                "parameters": unavailable("opaque state blob"),
            },
        )
    )
    vox_note = TrackState(id="cubase:track-vox", name="Lead Vox", track_type="audio")
    vox_note.field_provenance["role"] = annotation("user says this is the lead vocal")

    state = SessionState(
        project=ProjectMeta(project_name="unit", project_path="/tmp/unit.dawproject"),
        tracks=[kick, vox_note],
        folders=[
            FolderState(
                id="cubase:folder-drums", name="Drums",
                child_track_ids=["cubase:track-kick"],
                organizational_only=True, group_channel_enabled=False,
            )
        ],
        unknown_state=[
            UnknownState(
                id="cubase:gap-plugin-parameters", entity_id=None,
                state_gap="plugin_parameters", reason="opaque blobs",
            ),
            UnknownState(
                id="cubase:gap-param-eq", entity_id="cubase:device-eq",
                state_gap="insert_parameter_state", reason="no surface exposes it",
            ),
        ],
    )
    return state


def test_mapper_folders_become_group_tracks():
    session = session_state_to_canonical(_make_state())
    folder = session.track_by_id("cubase:folder-drums")
    assert folder is not None
    assert folder.kind == "group"
    assert folder.extras["folder"] is True
    assert folder.extras["organizational_only"] is True
    assert folder.extras["group_channel_enabled"] is False
    assert folder.extras["child_track_ids"] == ["cubase:track-kick"]
    # containment is carried by the child's group_id
    kick = session.track_by_id("cubase:track-kick")
    assert kick.group_id == "cubase:folder-drums"


def test_mapper_provenance_status_vocabulary():
    session = session_state_to_canonical(_make_state())
    kick = session.track_by_id("cubase:track-kick")
    # exported -> observed
    assert kick.provenance.observability == "observed"
    assert kick.field_provenance["volume_db"].observability == "observed"
    # inferred -> inferred, confidence preserved
    assert kick.field_provenance["role"].observability == "inferred"
    assert kick.field_provenance["role"].confidence == 0.55
    # conflicting -> inferred with an explanation recording the conflict
    pan = kick.field_provenance["pan"]
    assert pan.observability == "inferred"
    assert "runtime observed" in pan.explanation
    # unavailable -> hidden
    eq = kick.processors[0]
    assert eq.field_provenance["parameters"].observability == "hidden"
    # user_supplied -> annotation
    vox = session.track_by_id("cubase:track-vox")
    assert vox.field_provenance["role"].observability == "annotation"


def test_mapper_unknown_state_becomes_hidden_markers():
    session = session_state_to_canonical(_make_state())
    markers = {m.id: m for m in session.hidden_state_markers}
    assert len(markers) == 2
    # un-anchored gaps target the flat PROJECT id convention
    project_gap = markers["cubase:gap-plugin-parameters"]
    assert project_gap.target_id == "cubase:project"
    assert project_gap.hidden_state_type == "plugin_parameters"
    # STATE_GAP_INFO enriches consequence + possible sources
    assert "preset" in " ".join(project_gap.possible_sources).lower()
    # entity-anchored gaps keep their anchor
    device_gap = markers["cubase:gap-param-eq"]
    assert device_gap.target_id == "cubase:device-eq"


def test_mapper_rejects_foreign_native_models():
    with pytest.raises(TypeError, match="SessionState only"):
        to_canonical({"not": "a session"})  # type: ignore[arg-type]


def test_native_payload_round_trip():
    state = _make_state()
    session = session_state_to_canonical(state)
    assert to_native(session) == state


def test_mapper_emits_automation_from_demo_dawproject(fixtures_dir):
    """The demo dawproject carries a real vocal-Volume lane; ``to_canonical``
    must forward it into ``session.automation`` (in addition to — not instead
    of — the back-compat extras summary)."""
    from cubase_session_explorer.fusion import ingest

    src = os.path.join(fixtures_dir, "demo_session.dawproject")
    state = ingest(src, hash_files=False).session
    assert state.automation, "demo dawproject should carry an automation lane"

    session = to_canonical(state)
    assert session.automation, "to_canonical must populate session.automation"

    lane = next(a for a in session.automation if a.parameter_name == "Volume")
    assert lane.unit == "linear"
    assert len(lane.points) == 3
    assert [p.value for p in lane.points] == [0.6, 0.8, 0.7]
    assert all(p.time_domain == "beats" for p in lane.points)
    # Channel-strip Volume automation → mixer field, not a device parameter.
    assert lane.target_processor_id is None
    assert lane.target_channel_field == "volume"
    assert lane.target_track_id == "cubase:track-tr-ch-vox"
    assert lane.target_parameter_id == "cubase:ch-vox-vol"
    # DAWproject is an official export → OBSERVED.
    assert lane.provenance.observability == "observed"
    # Back-compat: the extras summary is still emitted alongside the model.
    assert session.extras.get("automation_lanes")


# ---------------------------------------------------------------------------
# bundle: .dawproject (full-strength pathway)
# ---------------------------------------------------------------------------


@pytest.fixture()
def demo_bundle(fixtures_dir, tmp_path):
    src = os.path.join(fixtures_dir, "demo_session.dawproject")
    result = exporter.export_bundle(src, str(tmp_path / "bundle"))
    return result


def _load(result, name):
    with open(result.files[name], encoding="utf-8") as fh:
        return json.load(fh)


def test_dawproject_bundle_has_five_valid_files(demo_bundle):
    assert set(demo_bundle.files) == BUNDLE_FILES
    validation = _load(demo_bundle, "validation.json")
    assert validation["valid"] is True
    assert validation["errors"] == []
    # what is on disk re-validates independently
    snapshot_dict = _load(demo_bundle, "canonical.snapshot.json")
    report = validate_snapshot(snapshot_dict)
    assert report.valid


def test_dawproject_bundle_track_channel_split(demo_bundle):
    snap = demo_bundle.snapshot
    tracks = snap.entities_of_type("TRACK")
    channels = snap.entities_of_type("CHANNEL")
    assert tracks and channels
    # TRACK != CHANNEL: return/master are channel-only, so the counts differ
    assert len(tracks) != len(channels)
    links = snap.relationships_of_type("TRACK_USES_CHANNEL")
    assert {r.source for r in links} == {t.id for t in tracks}
    # channel state landed on the CHANNEL side of the split
    kick_channel = next(c for c in channels if c.name == "Kick")
    assert "volume_db" in kick_channel.properties
    assert "pan" in kick_channel.properties
    assert "mute" in kick_channel.properties
    # processing order is explicit on the edge
    processed = snap.relationships_of_type("CHANNEL_PROCESSED_BY")
    assert processed and all("index" in r.properties for r in processed)
    # send level rides the routing edge
    sends = snap.relationships_of_type("CHANNEL_SENDS_TO")
    assert sends and any("volume_db" in r.properties for r in sends)


def test_dawproject_bundle_automation_controls_channel(demo_bundle):
    """The real vocal-Volume lane becomes an AUTOMATION entity whose CONTROLS
    edge targets the vocal channel's volume field."""
    snap = demo_bundle.snapshot
    autos = snap.entities_of_type("AUTOMATION")
    assert len(autos) == 1
    auto = autos[0]
    assert auto.properties["unit"] == "linear"
    assert auto.properties["point_count"] == 3
    assert auto.availability == {}  # target resolved, not an UNKNOWN

    controls = snap.relationships_of_type("CONTROLS")
    assert len(controls) == 1
    edge = controls[0]
    assert edge.source == auto.id
    assert edge.target == "cubase:track-tr-ch-vox:channel"
    assert edge.properties.get("field") == "volume"

    # snapshot-level automation array + coverage both reflect the lane
    assert len(snap.automation) == 1
    assert snap.coverage["automation"].observed == 1


def test_dawproject_bundle_provenance_stability_and_evidence(demo_bundle):
    snap = demo_bundle.snapshot
    stabilities = {p.source_stability for p in snap.provenance}
    assert stabilities == {"OFFICIAL_EXPORT"}
    evidence = {p.evidence for p in snap.provenance}
    # exported facts, inferred heuristics (role/family backfill), and the
    # hidden plug-in-parameter state must all be represented
    assert {"OBSERVED", "INFERRED", "HIDDEN"} <= evidence
    # confidence never rides on OBSERVED records
    assert all(p.confidence is None for p in snap.provenance if p.evidence == "OBSERVED")
    # every prov ref resolves (validation checks this too; belt and braces)
    prov_ids = {p.id for p in snap.provenance}
    for entity in snap.entities:
        assert set(entity.prov.values()) <= prov_ids


def test_dawproject_bundle_hidden_state_lands_as_availability(demo_bundle):
    snap = demo_bundle.snapshot
    project = snap.entity_by_id("cubase:project")
    # un-anchored observation-model gaps attach to the PROJECT entity
    assert project.availability.get("plugin_parameters") == "INACCESSIBLE"
    # device-anchored parameter gaps attach to the PROCESSOR entities
    processors = snap.entities_of_type("PROCESSOR")
    assert processors
    assert any(
        e.availability.get("insert_parameter_state") == "INACCESSIBLE"
        for e in processors
    )


def test_dawproject_bundle_native_ref_and_sanitization(demo_bundle):
    snap = demo_bundle.snapshot
    ref = snap.extensions["cubase"]["native_file"]
    assert ref["path"] == "native.json"
    with open(demo_bundle.files["native.json"], "rb") as fh:
        payload = fh.read()
    from cubase_session_explorer.utils import sha256_bytes

    assert ref["sha256"] == sha256_bytes(payload)
    native = json.loads(payload)
    assert native["model_name"] == "SessionState"
    # the native payload re-validates as the native model (losslessness anchor)
    from cubase_session_explorer.models import SessionState as NativeSessionState

    NativeSessionState.model_validate(native["model"])
    # sanitization: no home directory or fixture-directory absolute paths
    home = os.path.expanduser("~")
    for name in ("canonical.snapshot.json", "native.json"):
        text = open(demo_bundle.files[name], encoding="utf-8").read()
        assert home not in text
        assert "/fixtures/cubase/" not in text


def test_dawproject_bundle_is_deterministic(fixtures_dir, tmp_path):
    src = os.path.join(fixtures_dir, "demo_session.dawproject")
    a = exporter.export_bundle(src, str(tmp_path / "a"))
    b = exporter.export_bundle(src, str(tmp_path / "b"))
    for name in ("canonical.snapshot.json", "native.json"):
        assert open(a.files[name], "rb").read() == open(b.files[name], "rb").read()


def test_capability_manifest_shape(demo_bundle):
    caps = _load(demo_bundle, "capabilities.json")
    assert caps["daw"] == "cubase"
    assert caps["adapter"] == "cubase-hybrid"
    # write and render claim nothing
    assert caps["write"] == {}
    assert caps["render"] == {}
    # dawproject read pathway is OFFICIAL_EXPORT; TESTED only where fixtures cover it
    structure = caps["read"]["structure"]["fields"]
    assert structure["track_name"]["source_stability"] == "OFFICIAL_EXPORT"
    assert structure["track_name"]["validation_status"] == "TESTED"
    assert structure["folder_group_channel"]["validation_status"] != "TESTED"
    # plug-in parameters are PARTIAL: built-in devices enumerate host-visible
    # parameters; third-party state blobs stay opaque
    processing = caps["read"]["processing"]["fields"]
    assert processing["plugin_parameters"]["support"] == "PARTIAL"
    # cpr is reverse-engineered evidence-only
    cpr = caps["read"]["cpr_evidence"]["fields"]
    assert cpr["structural_state"]["support"] == "NONE"
    assert cpr["plugin_name_candidates"]["source_stability"] == "REVERSE_ENGINEERED"
    # the runtime bridge exists but is not validated here
    live = caps["live_observation"]["channel"]["fields"]
    assert all(f["validation_status"] in ("UNTESTED", "CLAIMED") for f in live.values())
    descriptor = _load(demo_bundle, "adapter_descriptor.json")
    assert descriptor["adapter_id"] == "cubase-hybrid"
    assert descriptor["write"] == "NONE"


# ---------------------------------------------------------------------------
# bundle: .cpr (degraded but honest)
# ---------------------------------------------------------------------------


def test_cpr_bundle_is_degraded_but_valid(tmp_path):
    blob = (b"RIFF????NUNDROOT....CmObject...DualFilter\x00\x00StudioEQ.."
            b"PAppVersion 15.0")
    src = tmp_path / "fake.cpr"
    src.write_bytes(blob)
    result = exporter.export_bundle(str(src), str(tmp_path / "bundle"))

    assert set(result.files) == BUNDLE_FILES
    assert result.validation.valid is True

    snap = result.snapshot
    # exactly one entity: the PROJECT — nothing structural is fabricated
    assert [e.entity_type for e in snap.entities] == ["PROJECT"]
    project = snap.entities[0]
    assert project.properties["is_probable_cpr"] is True
    # the unreadable state is an explicit availability ledger, not silence
    assert project.availability["structure"] == "INACCESSIBLE"
    assert project.availability["routing"] == "INACCESSIBLE"
    # the failure is shipped, not hidden
    assert snap.failures and snap.failures[0].stage == "structural_parse"
    assert "UNSUPPORTED" in snap.failures[0].message
    # the full evidence report rides in the namespaced extension
    report = snap.extensions["cubase"]["cpr_report"]
    assert report["is_probable_cpr"] is True
    assert any("DualFilter" in v for v in
               [e["value"] for e in report["plugin_name_candidates"]])
    assert snap.source.capture_modes == ["cpr_evidence_scan"]
    assert {p.source_stability for p in snap.provenance} == {"REVERSE_ENGINEERED"}


def test_cpr_bundle_unrecognized_container_is_unknown(tmp_path):
    src = tmp_path / "mystery.cpr"
    src.write_bytes(b"\x00" * 64)
    result = exporter.export_bundle(str(src), str(tmp_path / "bundle"))
    assert result.validation.valid is True
    project = result.snapshot.entities[0]
    assert project.properties["is_probable_cpr"] is False
    # not even recognizably a CPR: we do not know what is in there
    assert project.availability["structure"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# bundle: .mid (evidence-only musical content)
# ---------------------------------------------------------------------------


def test_midi_bundle_is_evidence_only(fixtures_dir, tmp_path):
    src = os.path.join(fixtures_dir, "notes.mid")
    result = exporter.export_bundle(src, str(tmp_path / "bundle"))

    assert set(result.files) == BUNDLE_FILES
    assert result.validation.valid is True

    snap = result.snapshot
    types = sorted({e.entity_type for e in snap.entities})
    assert types == ["MUSICAL_CONTENT", "PROJECT"]
    project = snap.entity_by_id("cubase:project")
    assert project.properties["tempo"] == 120.0
    # this surface cannot carry structure/routing/processing — said explicitly
    assert project.availability["structure"] == "UNSUPPORTED"
    assert project.availability["routing"] == "UNSUPPORTED"
    content = snap.entities_of_type("MUSICAL_CONTENT")
    assert len(content) == 1
    assert content[0].properties["note_count"] == 4
    assert content[0].properties["pitch_min"] == 60
    # containment edges tie content to the project
    contains = snap.relationships_of_type("CONTAINS")
    assert {(r.source, r.target) for r in contains} == {
        (project.id, content[0].id)
    }
    assert {p.source_stability for p in snap.provenance} == {"OFFICIAL_DOCUMENTED"}
    assert {p.evidence for p in snap.provenance} == {"OBSERVED"}


# ---------------------------------------------------------------------------
# dispatch honesty
# ---------------------------------------------------------------------------


def test_unsupported_suffix_fails_loudly(tmp_path):
    src = tmp_path / "session.als"
    src.write_bytes(b"not ours")
    with pytest.raises(exporter.ExportError, match="Unsupported input suffix"):
        exporter.export_bundle(str(src), str(tmp_path / "bundle"))


def test_missing_input_fails_loudly(tmp_path):
    with pytest.raises(exporter.ExportError, match="not found"):
        exporter.export_bundle(str(tmp_path / "nope.mid"), str(tmp_path / "bundle"))
