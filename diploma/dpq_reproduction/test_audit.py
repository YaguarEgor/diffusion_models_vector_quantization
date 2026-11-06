import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name) / 'upstream'
        self.repo.mkdir()
        for name in ('pqf/compressed_layers/CompressedLinear.py', 'pq/utils_traineval.py',
                     'models.py', 'pq/utils_model.py', 'pqf/utils/model_size.py'):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# source\n', encoding='utf-8')
        self.hashes = {p.relative_to(self.repo).as_posix(): audit.hash_file(p)
                       for p in self.repo.rglob('*.py')}
        self.manifest = self.repo.parent / 'upstream-manifest.json'
        with patch.object(audit, 'git_inventory', return_value=self.hashes):
            audit.seal(self.repo)

    def inspect_without_git(self):
        with patch.object(audit, 'git_inventory', side_effect=FileNotFoundError('git')):
            return audit.inspect(self.repo)

    def test_no_git_uses_verified_seal(self):
        report = self.inspect_without_git()
        self.assertEqual(report['commit'], audit.COMMIT)
        self.assertEqual(report['python_source_sha256'], self.hashes)

    def test_tampered_source_rejected(self):
        (self.repo / 'models.py').write_text('# changed\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.inspect_without_git()

    def test_added_source_rejected(self):
        (self.repo / 'extra.py').write_text('# injected\n', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.inspect_without_git()

    def test_missing_seal_rejected(self):
        self.manifest.unlink()
        with self.assertRaises(FileNotFoundError):
            self.inspect_without_git()

    def test_wrong_commit_rejected(self):
        report = json.loads(self.manifest.read_text())
        report['commit'] = 'wrong'
        self.manifest.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.inspect_without_git()

    def test_path_escape_rejected(self):
        report = json.loads(self.manifest.read_text())
        report['tracked_sha256']['../outside.py'] = '0' * 64
        self.manifest.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.inspect_without_git()
