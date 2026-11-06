import importlib.util
import json
from pathlib import Path
import unittest

import torch
from torch import nn

from paper_core import PaperLinear,block_loss,nearest_candidates,storage

HERE=Path(__file__).resolve().parent


class PaperMathTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.book=torch.tensor([[0.,0.],[1.,2.],[-1.,1.]])
        self.candidates=torch.tensor([[0,1],[2,0]],dtype=torch.uint8)

    def layer(self):
        return PaperLinear(self.book,self.candidates,(2,2),torch.zeros(2))

    def test_initial_equal_ratios_and_average(self):
        m=self.layer()
        self.assertTrue(torch.equal(m.logits.softmax(-1),torch.full((2,2),.5)))
        self.assertTrue(torch.equal(m.weight(),torch.stack((self.book[:2].mean(0),self.book[[2,0]].mean(0)))))

    def test_polarization_equation9(self):
        m=self.layer()
        self.assertEqual(m.polarization().item(),2.)
        with torch.no_grad():
            m.logits.copy_(torch.tensor([[1.,-1.],[2.,-2.]]))
        p=m.logits.softmax(-1)
        expected=sum(1-abs(2*p[i,j]-1) for i in range(2) for j in range(2))/2
        torch.testing.assert_close(m.polarization(),expected)
        m.polarization().backward()
        self.assertGreater(m.logits.grad.norm().item(),0)

    def test_gradient_reaches_books_and_ratios(self):
        m=self.layer()
        x=torch.tensor([[1.,3.],[2.,-1.]])
        (m(x)-torch.tensor([[2.,0.],[1.,4.]])).square().mean().backward()
        for p in (m.book,m.logits,m.bias):
            self.assertIsNotNone(p.grad)
            self.assertGreater(p.grad.norm().item(),0)

    def test_argmax_not_candidate_average_export(self):
        m=self.layer()
        with torch.no_grad():
            m.logits.copy_(torch.tensor([[-2.,3.],[2.,-1.]]))
        exported=m.exported()
        self.assertTrue(torch.equal(exported['codes'],torch.tensor([1,2],dtype=torch.uint8)))
        m.harden()
        self.assertFalse(m.logits.requires_grad)
        self.assertTrue(torch.equal(m.weight(),self.book[[1,2]]))
        self.assertEqual(m.polarization().item(),0)
        m.harden()

    def test_candidates_chunking_and_exact_distance(self):
        x=torch.tensor([[.1,.2],[1.,1.9],[-.9,.9],[2.,3.]])
        expected=((x[:,None]-self.book[None])**2).sum(-1).topk(2,largest=False).indices.byte()
        for chunk in (1,3,100):
            self.assertTrue(torch.equal(nearest_candidates(x,self.book,chunk=chunk),expected))

    def test_invalid_indices_and_shapes(self):
        for candidates,shape in [(torch.tensor([[0,0],[1,2]],dtype=torch.uint8),(2,2)),
                                  (torch.tensor([[0,3],[1,2]],dtype=torch.uint8),(2,2)),
                                  (self.candidates,(2,3))]:
            with self.assertRaises(ValueError):
                PaperLinear(self.book,candidates,shape)

    def test_block_loss_is_sum_of_block_means(self):
        a=torch.tensor([[1.,3.]],requires_grad=True)
        b=torch.tensor([[2.,4.]],requires_grad=True)
        actual=block_loss([a,b],[torch.zeros_like(a),torch.zeros_like(b)])
        self.assertEqual(actual.item(),15.)
        actual.backward()
        torch.testing.assert_close(a.grad,a.detach())

    def test_microbatch_same_gradient(self):
        full=self.layer();split=self.layer()
        x=torch.randn(8,2);y=torch.randn(8,2)
        ((full(x)-y).square().mean()+full.polarization()).backward()
        for i in (0,4):
            ((split(x[i:i+4])-y[i:i+4]).square().mean()/2).backward()
        split.polarization().backward()
        for a,b in zip(full.parameters(),split.parameters()):
            torch.testing.assert_close(a.grad,b.grad)

    def test_storage_excludes_training_candidates_and_ratios(self):
        m=self.layer();stats=storage([('a',m)],10)
        expected=2*8+6*32+2*32+10*32
        self.assertEqual(stats['logical_bits'],expected)
        self.assertEqual(stats['whole_dit_bpp'],expected/(4+2+10))

    def test_author_lloyd_mean(self):
        spec=importlib.util.spec_from_file_location('author_kmeans_test',HERE/'upstream/kmeans.py')
        author=importlib.util.module_from_spec(spec);spec.loader.exec_module(author)
        x=torch.tensor([[0.,0.],[1.,1.],[9.,9.],[10.,10.]])
        codes=torch.tensor([0,0,1,1]);book=torch.empty(2,2)
        author.update_codebook(codes,x,book)
        self.assertTrue(torch.equal(book,torch.tensor([[.5,.5],[9.5,9.5]])))

    def test_fixed_paper_settings(self):
        cfg=json.loads((HERE/'paper_config.json').read_text())
        for key,value in dict(updates=500,batch_images=16,k=256,d=4,candidates=2,
                              ratios_lr=.05,codebook_and_block_bias_lr=.0001,cfg_scale=1.5,
                              evaluation_count=10000).items():
            self.assertEqual(cfg[key],value)
        self.assertEqual(cfg['initial_ratios'],[.5,.5])
        self.assertEqual(cfg['evaluation_steps'],[50])
        self.assertEqual(cfg['vae'],'stabilityai/sd-vae-ft-ema')
        self.assertEqual(len(cfg['vae_revision']),40)


if __name__=='__main__':
    unittest.main()
