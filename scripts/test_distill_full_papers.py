import unittest
from distill_full_papers import validate

class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.note = dict(paper_id='1234.56789',title='Fixture',contribution='Fixture',
            architecture={k:[] for k in ('inputs','blocks','connections','outputs')},
            training={k:'not specified' for k in ('objective','optimizer','data','hyperparameters','evaluation')},
            component_mappings=[dict(component_id='Dense',kind='operator',role='head',limitation='not whole architecture')],
            capability_gaps=['unknown'],implementation_steps=['inspect'],verification_steps=['test'],
            source_anchors=[dict(section='3',quote='exact source phrase')],limitations=['review needed'])
        self.registry={'components':[dict(id='Dense',kind='operator')]}
    def check(self,note):
        return validate(note,'1234.56789','This is an exact source phrase.',self.registry)
    def test_valid(self):
        self.check(self.note)
    def test_normalizes_single_anchor(self):
        self.note['source_anchors']=self.note['source_anchors'][0]
        checked=self.check(self.note)
        self.assertEqual(len(checked['source_anchors']),1)
    def test_rejects_identity(self):
        self.note['paper_id']='other'
        with self.assertRaises(ValueError): self.check(self.note)
    def test_rejects_invented_component(self):
        self.note['component_mappings'][0]['component_id']='Imagined'
        with self.assertRaises(ValueError): self.check(self.note)
    def test_rejects_invented_quote(self):
        self.note['source_anchors'][0]['quote']='not in paper'
        with self.assertRaises(ValueError): self.check(self.note)
    def test_rejects_missing_recipe(self):
        del self.note['training']['objective']
        with self.assertRaises(ValueError): self.check(self.note)

if __name__=='__main__': unittest.main()
