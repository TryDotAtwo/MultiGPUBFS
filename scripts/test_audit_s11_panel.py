import unittest

import audit_s11_panel as panel


class SourcePinTests(unittest.TestCase):
    def test_explicit_source_is_accepted(self):
        panel.validate_environment({
            'source': '013ed5c979f4225db273e0015fa9ed72fd230c90',
            'baseline': 'f0f2b8e5ee61173039ab9742f3a7756c9b6365e6',
        }, '013ed5c979f4225db273e0015fa9ed72fd230c90')

    def test_wrong_source_is_rejected(self):
        with self.assertRaises(ValueError):
            panel.validate_environment({
                'source': '66d82d03cb055daa08dae328978208efda7c8ede',
                'baseline': 'f0f2b8e5ee61173039ab9742f3a7756c9b6365e6',
            }, '013ed5c979f4225db273e0015fa9ed72fd230c90')

    def test_wrong_baseline_is_rejected(self):
        with self.assertRaises(ValueError):
            panel.validate_environment({'source': 'a' * 40, 'baseline': 'b' * 40}, 'a' * 40)

    def test_unpinned_revision_is_rejected_even_when_equal(self):
        with self.assertRaises(ValueError):
            panel.validate_environment({
                'source': 'main',
                'baseline': 'f0f2b8e5ee61173039ab9742f3a7756c9b6365e6',
            }, 'main')


if __name__ == '__main__':
    unittest.main()
