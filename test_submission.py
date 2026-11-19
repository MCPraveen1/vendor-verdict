from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import submission


class FakeClient:
    def __init__(self, payload: dict):
        self.payload = payload
        self.request = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.request = kwargs
        message = SimpleNamespace(content=json.dumps(self.payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.case = {
            "case_id": "c9999",
            "vendor": "Example Vendor",
            "legal_entity": "Example Vendor Ltd",
            "official_domain": "example.test",
            "tier": "T3",
            "users": "100",
            "as_of": "2026-11-01",
        }
        self.official_page = {
            "case_id": "c9999",
            "page_id": "c9999-p01",
            "source_type": "official",
            "publisher": "Example Vendor",
            "title": "Privacy and storage",
            "url": "https://docs.example.test/privacy",
            "last_updated": "2026-10-01",
            "text": "Example Vendor Ltd stores customer data in India for active workspaces.",
        }

    def test_official_domain_and_exact_entity_required(self):
        self.assertEqual(submission.validate_page_identity(self.official_page, self.case), "CONFIRMED_MATCH")
        other = dict(self.official_page, url="https://example.test.attacker.invalid/privacy")
        self.assertEqual(submission.validate_page_identity(other, self.case), "MANUAL_REVIEW")
        ambiguous = dict(self.official_page, text="Example Vendor Ltd works with Other Company GmbH.")
        self.assertEqual(submission.validate_page_identity(ambiguous, self.case), "MANUAL_REVIEW")

    def test_citations_must_match_exact_source_and_identity(self):
        page = dict(self.official_page, identity_status="CONFIRMED_MATCH")
        citations = [{"page_id": page["page_id"], "statement": "Example Vendor Ltd stores customer data in India for active workspaces."}]
        value = {"available": ["India"], "planned": []}
        self.assertEqual(submission.validate_fact_value("data_regions", value, citations, [page]), value)
        bad = [{"page_id": page["page_id"], "statement": "Example Vendor stores all customer data in India."}]
        self.assertIsNone(submission.validate_fact_value("data_regions", value, bad, [page]))

    def test_groq_call_uses_configured_model_and_json_mode(self):
        payload = {"facts": {"data_regions": {"available": ["India"], "planned": []}}, "evidence": {"data_regions": [{"page_id": "c9999-p01", "statement": self.official_page["text"]}]}}
        fake = FakeClient(payload)
        page = dict(self.official_page, identity_status="CONFIRMED_MATCH")
        result = submission.call_groq(fake, self.case, [page], ["data_regions"])
        self.assertEqual(result, payload)
        self.assertEqual(fake.request["model"], "openai/gpt-oss-120b")
        self.assertEqual(fake.request["response_format"], {"type": "json_object"})

    def test_certification_requires_matching_official_and_registry_sources(self):
        official_text = "Example Vendor Ltd holds ISO/IEC 27001 certificate ISO-27001-5001."
        registry_text = (
            "Registry entry ISO-27001-5001 (holder: Example Vendor Ltd; "
            "standard: ISO/IEC 27001; certification body: Veridian Certification; "
            "issued 2025-09-27; expires 2027-10-01)."
        )
        pages = [
            dict(self.official_page, text=official_text, identity_status="CONFIRMED_MATCH"),
            {
                "case_id": "c9999", "page_id": "c9999-p02", "source_type": "regulator",
                "publisher": "Veridian Certification", "title": "Certificate ISO-27001-5001",
                "url": "https://registry.test/cert/5001", "text": registry_text,
                "identity_status": "REGISTRY_HOLDER_MATCH",
            },
        ]
        value = [{
            "name": "ISO 27001", "status": "certified", "date": "2025-09-27",
            "valid_until": "2027-10-01", "issuer": "Veridian Certification",
            "certificate_id": "ISO-27001-5001",
        }]
        citations = [
            {"page_id": "c9999-p01", "statement": official_text},
            {"page_id": "c9999-p02", "statement": registry_text},
        ]
        accepted = submission.validate_fact_value("security_certifications", value, citations, pages)
        self.assertEqual(accepted[0]["evidence_page_ids"], ["c9999-p02", "c9999-p01"])
        mismatch = dict(
            pages[1],
            text=registry_text.replace("Example Vendor Ltd", "Other Vendor Inc"),
            identity_status="REGISTRY_HOLDER_MISMATCH",
        )
        self.assertIsNone(submission.validate_fact_value("security_certifications", value, citations, [pages[0], mismatch]))

    def test_submission_order_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "submission.csv"
            rows = [{"id": "x1", "verdict": "UNKNOWN"}, {"id": "x2", "verdict": "CONDITIONAL"}]
            submission.write_submission(rows, output, ["x1", "x2"])
            self.assertEqual(output.read_text(encoding="utf-8").splitlines()[0], "id,verdict")
            with self.assertRaises(FileExistsError):
                submission.write_submission(rows, output, ["x1", "x2"])
            with self.assertRaises(ValueError):
                submission.write_submission(rows[::-1], Path(directory) / "bad.csv", ["x1", "x2"])

    def test_competition_files_and_coverage(self):
        cases, pages, requirements, test_rows, train_rows = submission.load_inputs(Path(__file__).resolve().parent)
        self.assertEqual(len(cases), 700)
        self.assertEqual(len(pages), 7396)
        self.assertEqual(len(test_rows), 2378)
        self.assertEqual(len(train_rows), 3185)
        self.assertEqual({req["field"] for req in requirements}, set(submission.FACT_FIELDS))

    def test_train_run_writes_and_scores_complete_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {
                "case_id": "c0001", "split": "train", "vendor": "Example",
                "legal_entity": "Example Ltd", "official_domain": "example.test",
                "tier": "T3", "users": "10", "as_of": "2026-11-01",
            }
            (root / "cases.csv").write_text(
                "case_id,split,vendor,legal_entity,official_domain,tier,users,as_of\n"
                "c0001,train,Example,Example Ltd,example.test,T3,10,2026-11-01\n",
                encoding="utf-8",
            )
            (root / "pages.jsonl").write_text("", encoding="utf-8")
            requirement = [{
                "id": "R1", "field": "data_regions", "rule": "value_known",
                "severity": "major", "params": {},
                "applies_when": {"tiers": ["T3"], "or_min_users": None},
            }]
            (root / "requirements.json").write_text(json.dumps(requirement), encoding="utf-8")
            expected = ["c0001__R1", "c0001__OVERALL"]
            (root / "test.csv").write_text(
                "id,case_id,requirement_id\nc0001__R1,c0001,R1\nc0001__OVERALL,c0001,OVERALL\n",
                encoding="utf-8",
            )
            (root / "sample_submission.csv").write_text(
                "id,verdict\nc0001__R1,UNKNOWN\nc0001__OVERALL,CONDITIONAL\n",
                encoding="utf-8",
            )
            (root / "train_labels.csv").write_text(
                "id,case_id,requirement_id,verdict\nc0001__R1,c0001,R1,UNKNOWN\nc0001__OVERALL,c0001,OVERALL,CONDITIONAL\n",
                encoding="utf-8",
            )
            result = submission.run(
                root, root / "train_predictions.csv", root / "cache.json",
                split="train", mode="offline",
            )
            self.assertEqual(result["rows"], 2)
            self.assertEqual(result["evaluation"]["score"], 1.0)
            self.assertEqual(
                (root / "train_predictions.csv").read_text(encoding="utf-8").splitlines()[1:],
                ["c0001__R1,UNKNOWN", "c0001__OVERALL,CONDITIONAL"],
            )

    def test_groq_pipeline_end_to_end_with_mock_response_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = {
                "case_id": "c0001", "split": "test", "vendor": "Example",
                "legal_entity": "Example Vendor Ltd", "official_domain": "example.test",
                "tier": "T3", "users": "10", "as_of": "2026-11-01",
            }
            (root / "cases.csv").write_text(
                "case_id,split,vendor,legal_entity,official_domain,tier,users,as_of\n"
                "c0001,test,Example,Example Vendor Ltd,example.test,T3,10,2026-11-01\n",
                encoding="utf-8",
            )
            statement = "Example Vendor Ltd customer data stored in India."
            page = {
                "case_id": "c0001", "page_id": "c0001-p01", "source_type": "official",
                "publisher": "Example", "title": "Data storage region",
                "url": "https://example.test/privacy", "last_updated": "2026-10-01",
                "text": statement,
            }
            (root / "pages.jsonl").write_text(json.dumps(page) + "\n", encoding="utf-8")
            (root / "requirements.json").write_text(json.dumps([{
                "id": "R1", "field": "data_regions", "rule": "region_includes",
                "severity": "blocker", "params": {"any_of": ["India"], "allow_planned": False},
                "applies_when": {"tiers": ["T3"], "or_min_users": None},
            }]), encoding="utf-8")
            ids = ["c0001__R1", "c0001__OVERALL"]
            (root / "test.csv").write_text(
                "id,case_id,requirement_id\nc0001__R1,c0001,R1\nc0001__OVERALL,c0001,OVERALL\n",
                encoding="utf-8",
            )
            (root / "sample_submission.csv").write_text(
                "id,verdict\nc0001__R1,UNKNOWN\nc0001__OVERALL,CONDITIONAL\n",
                encoding="utf-8",
            )
            (root / "train_labels.csv").write_text(
                "id,case_id,requirement_id,verdict\n", encoding="utf-8"
            )
            fake = FakeClient({
                "facts": {"data_regions": {"available": ["India"], "planned": []}},
                "evidence": {"data_regions": [{"page_id": "c0001-p01", "statement": statement}]},
            })
            result = submission.run(
                root, root / "mock_submission.csv", root / "cache.json",
                split="test", mode="groq", client=fake,
            )
            with (root / "mock_submission.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(result["rows"], 2)
            self.assertEqual([row["id"] for row in rows], ids)
            self.assertEqual([row["verdict"] for row in rows], ["PASS", "READY_FOR_APPROVAL"])
            self.assertEqual(fake.request["model"], "openai/gpt-oss-120b")


if __name__ == "__main__":
    unittest.main()
