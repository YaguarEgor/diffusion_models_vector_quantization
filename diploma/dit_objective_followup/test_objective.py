"""CPU checks for layout, deployment rounding and the new gradient paths."""
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
PINNED = HERE.parent/'dit_loss_recode'
if not PINNED.exists():
    PINNED = HERE.parent/'dit_loss_recode_20261027'
sys.path.insert(0,str(PINNED))
sys.path.insert(0,str(HERE))
from core import AdditiveLinear,dequantize
from objective_core import FollowupLinear,hard_assignments,nearest_alternatives,polarization,ddpm_objective


def values():
    torch.manual_seed(7)
    return dict(books=torch.randn(1,2,4,1,2)*.1,codes=torch.randint(4,(3,4,2)).int(),
                scales=torch.ones(3,1,1,1).half(),bias=torch.randn(3).half())


class Tests(unittest.TestCase):
    def test_original_reconstruction_exact(self):
        v=values()
        a=AdditiveLinear(**v)
        b=FollowupLinear(v)
        self.assertTrue(torch.equal(a.weight(),b.weight()))

    def test_scale_zero_is_exact(self):
        v=values()
        m=FollowupLinear(v,train_scales=True)
        self.assertTrue(torch.equal(m.weight(),AdditiveLinear(**v).weight()))
        m(torch.randn(5,8)).square().mean().backward()
        self.assertIsNotNone(m.log_scale.grad)
        self.assertGreater(m.log_scale.grad.abs().sum().item(),0)

    def test_scale_export_rounds_to_actual_storage(self):
        v=values();m=FollowupLinear(v,train_scales=True)
        with torch.no_grad():m.log_scale.fill_(.13)
        e=m.exported()
        self.assertTrue(torch.equal(m.weight(),AdditiveLinear(**e).weight()))
        self.assertEqual(e['scales'].dtype,torch.float16)
        self.assertEqual(e['codes'].dtype,torch.uint8)
        self.assertEqual(set(e),{'books','codes','scales','bias'})

    def test_hard_assignment_current_at_start(self):
        v=values();alt=(v['codes']+1)%4
        m=FollowupLinear(v,alternatives=alt)
        with hard_assignments(m):
            self.assertTrue(torch.equal(m.weight(),AdditiveLinear(**v).weight()))
        self.assertFalse(m.hard)

    def test_soft_gradients_finite(self):
        v=values();m=FollowupLinear(v,alternatives=(v['codes']+1)%4)
        m(torch.randn(5,8)).square().mean().backward()
        for parameter in (m.books,m.logits):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(),0)

    def test_tiny_probability_gradients_do_not_underflow_fp16(self):
        v=values();m=FollowupLinear(v,alternatives=(v['codes']+1)%4)
        (1e-8*m(torch.randn(5,8)).square().mean()).backward()
        self.assertGreater(m.logits.grad.abs().sum().item(),0)

    def test_hard_export_matches_actual_inference(self):
        v=values();m=FollowupLinear(v,alternatives=(v['codes']+1)%4)
        with torch.no_grad():m.logits[0].fill_(1)
        with hard_assignments(m):
            self.assertTrue(torch.equal(m.weight(),AdditiveLinear(**m.exported()).weight()))
        self.assertTrue(torch.equal(m.hard_codes()[0],m.alternatives[0]))
        self.assertTrue(torch.equal(m.hard_codes()[1:],m.codes[1:]))

    def test_ties_keep_initial_index(self):
        v=values();m=FollowupLinear(v,alternatives=(v['codes']+1)%4)
        with torch.no_grad():m.logits.zero_()
        self.assertTrue(torch.equal(m.hard_codes(),v['codes']))

    def test_hard_context_restores_on_error(self):
        m=FollowupLinear(values())
        with self.assertRaises(ValueError):
            with hard_assignments(m):
                self.assertTrue(m.hard)
                raise ValueError()
        self.assertFalse(m.hard)

    def test_nearest_alternative_matches_brute_force(self):
        v=values();reference=torch.randn(3,8)
        a=nearest_alternatives(v,reference,chunk=2)
        books=v['books'][0,:,:,0].float();codes=v['codes'].long()
        for o in range(3):
            for g in range(4):
                for s in range(2):
                    other=sum(books[k,codes[o,g,k]] for k in range(2) if k!=s)
                    target=reference[o,g*2:g*2+2]/v['scales'][o,0,0,0]-other
                    distances=(books[s]-target).square().sum(1)
                    distances[codes[o,g,s]]=float('inf')
                    self.assertEqual(a[o,g,s].item(),distances.argmin().item())
        self.assertTrue(torch.equal(a,nearest_alternatives(v,reference,chunk=10000)))

    def test_polarization_near_hard(self):
        v=values();m=FollowupLinear(v,alternatives=(v['codes']+1)%4)
        with torch.no_grad():m.logits.fill_(0)
        self.assertAlmostEqual(polarization(m).item(),2)
        with torch.no_grad():m.logits.fill_(-20)
        self.assertLess(polarization(m).item(),1e-6)

    def test_no_training_only_metadata_in_export(self):
        v=values();m=FollowupLinear(v,alternatives=(v['codes']+1)%4)
        self.assertEqual(sum(x.numel()*x.element_size() for x in m.exported().values()),
                         sum(x.numel()*x.element_size() for x in FollowupLinear(v).exported().values()))

    def test_variance_gradient_does_not_leak_into_epsilon(self):
        class Diffusion:
            def _vb_terms_bpd(self,model,**kwargs):
                p=model(None,None)
                return {'output':p.square().flatten(1).mean(1)}
        p=torch.randn(2,8,3,3,requires_grad=True)
        x=torch.zeros(2,4,3,3)
        loss,terms=ddpm_objective(Diffusion(),p,x,x,torch.tensor([2,4]),x)
        ge,=torch.autograd.grad(terms['variance_vb'],p,retain_graph=True)
        self.assertEqual(ge[:,:4].abs().sum().item(),0)
        self.assertGreater(ge[:,4:].abs().sum().item(),0)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())


if __name__=='__main__':unittest.main()
