#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes unitarios para kanon_weight_optimizer_v4.py (Fase 0).
Rodar:  python test_kanon_optimizer_v4.py
Cobre: AUC interno, StratifiedKFold, formula S=beta*S_text+(1-beta)*S_concepts,
       restricao SLSQP (soma=1, bounds), e analise de sensibilidade.
"""
import unittest
import numpy as np

import kanon_weight_optimizer_v4 as kx


class TestMetrics(unittest.TestCase):
    def test_roc_auc_perfect(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(kx.roc_auc(y, s), 1.0, places=6)

    def test_roc_auc_inverted(self):
        y = np.array([0, 0, 1, 1])
        s = np.array([0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(kx.roc_auc(y, s), 0.0, places=6)

    def test_roc_auc_ties(self):
        # empates -> 0.5 (rank medio)
        y = np.array([0, 1, 0, 1])
        s = np.array([0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(kx.roc_auc(y, s), 0.5, places=6)

    def test_roc_auc_single_class(self):
        self.assertTrue(np.isnan(kx.roc_auc(np.array([1, 1, 1]), np.array([1, 2, 3]))))


class TestKFold(unittest.TestCase):
    def test_stratified_balance_and_coverage(self):
        y = np.array([0] * 50 + [1] * 10)
        splits = kx.stratified_kfold_indices(y, 5, seed=42)
        self.assertEqual(len(splits), 5)
        all_test = np.concatenate([te for _, te in splits])
        # cada amostra aparece exatamente uma vez como teste
        self.assertEqual(sorted(all_test.tolist()), list(range(60)))
        for tr, te in splits:
            # cada fold de teste deve conter ao menos 1 Nobel (estratificacao)
            self.assertGreaterEqual(int(np.sum(y[te] == 1)), 1)
            # treino e teste disjuntos
            self.assertEqual(len(set(tr) & set(te)), 0)


class TestKanonFormula(unittest.TestCase):
    def setUp(self):
        self.comp = dict(
            C=np.array([0.5, 0.2]), A_pos=np.array([0.8, 0.4]),
            avg_auth=np.array([3.0, 10.0]),
            S_text=np.array([0.9, 0.1]), S_concepts=np.array([0.3, 0.7]),
            J=np.array([0.6, 0.2]))

    def test_S_is_convex_combination(self):
        # beta=1 => S=S_text ; beta=0 => S=S_concepts
        theta_b1 = [0.25, 0.25, 0.25, 0.25, 1.0, 1.0]
        theta_b0 = [0.25, 0.25, 0.25, 0.25, 0.0, 1.0]
        # Reconstroi S isolando wS: usa wS=1, demais 0
        only_s_b1 = kx.calc_kanon(self.comp, [0, 0, 1, 0, 1.0, 1.0]) / 100.0
        only_s_b0 = kx.calc_kanon(self.comp, [0, 0, 1, 0, 0.0, 1.0]) / 100.0
        np.testing.assert_allclose(only_s_b1, self.comp['S_text'], atol=1e-9)
        np.testing.assert_allclose(only_s_b0, self.comp['S_concepts'], atol=1e-9)

    def test_alpha_penalizes_hyperauthorship(self):
        # maior alpha => maior penalidade para muitos autores (A menor)
        low = kx.calc_kanon(self.comp, [0, 1, 0, 0, 0.5, 0.3]) / 100.0
        high = kx.calc_kanon(self.comp, [0, 1, 0, 0, 0.5, 1.0]) / 100.0
        # para avg_auth=10 (2o autor), A deve cair com alpha maior
        self.assertGreater(low[1], high[1])


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        n = 200
        # Sinal: Nobel tem C, S_text e J maiores
        y = np.array([0] * 160 + [1] * 40)
        base = rng.uniform(0, 0.4, size=n)
        boost = np.where(y == 1, 0.4, 0.0)
        self.comp = dict(
            C=np.clip(base + boost + rng.normal(0, 0.05, n), 0, 1),
            A_pos=rng.uniform(0.2, 0.9, n),
            avg_auth=rng.uniform(1, 8, n),
            S_text=np.clip(base + boost + rng.normal(0, 0.05, n), 0, 1),
            S_concepts=rng.uniform(0.5, 0.8, n),  # pouco discriminante (como no real)
            J=np.clip(base + boost * 0.5 + rng.normal(0, 0.05, n), 0, 1))
        self.y = y

    def test_constraint_and_bounds(self):
        out = kx.optimize_fold(self.comp, self.y, n_restarts=8, seed=1, max_weight=0.50)
        self.assertIsNotNone(out)
        theta = out['theta']
        w = theta[:4]
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)   # soma=1 imposta
        self.assertTrue(np.all(w >= 0.01 - 1e-6))               # bound inferior
        self.assertTrue(np.all(w <= 0.50 + 1e-6))               # bound superior (0.50)
        self.assertTrue(0.0 <= theta[4] <= 1.0)                 # beta
        self.assertTrue(0.3 <= theta[5] <= 1.0)                 # alpha

    def test_optimizer_finds_signal(self):
        out = kx.optimize_fold(self.comp, self.y, n_restarts=12, seed=2, max_weight=0.50)
        # AUC de treino deve ser substancialmente melhor que o acaso
        self.assertGreater(out['train_auc'], 0.75)


class TestSensitivity(unittest.TestCase):
    def test_sensitivity_keys(self):
        comp = dict(
            C=np.array([0.5, 0.2, 0.9, 0.1]), A_pos=np.array([0.8, 0.4, 0.7, 0.3]),
            avg_auth=np.array([3.0, 10.0, 2.0, 5.0]),
            S_text=np.array([0.9, 0.1, 0.8, 0.2]), S_concepts=np.array([0.3, 0.7, 0.6, 0.5]),
            J=np.array([0.6, 0.2, 0.7, 0.3]))
        y = np.array([1, 0, 1, 0])
        theta = np.array([0.25, 0.25, 0.25, 0.25, 0.5, 0.6])
        s = kx.sensitivity_analysis(comp, y, theta)
        for k in ['wC', 'wA', 'wS', 'wJ', 'beta+', 'beta-']:
            self.assertIn(k, s['components'])
        self.assertIn('base_auc', s)


if __name__ == '__main__':
    unittest.main(verbosity=2)
