import tensorflow as tf
import tf_metric_measures
import unittest

class MetricMeasuresTests(unittest.TestCase):
    def test_find_max_dq_index_expect_exclude_affine_return_first(self):
        p = tf.constant([ 1.+0j, 32+32j,  16.e-1+16.e-1j, 8.e-1+8.e-1j, \
            2.e-01+2.e-01j], dtype=tf.complex64)

        max_idx = tf_metric_measures.find_max_dq_coord_index(p)

        self.assertEqual(max_idx, 1)

    def test_good_coord_mask_expect_first_two_elements_false(self):
        p = tf.constant([ 1.+0j, 32+32j,  16.e-1+16.e-1j, 8.e-1+8.e-1j, \
            2.e-01+2.e-01j], dtype=tf.complex64)

        mask = tf_metric_measures.good_coord_mask(p)

        expect= tf.constant([ False, False, True, True, True])
        self.assertEqual(tf.reduce_all(tf.equal(mask, expect)), True)

if __name__ == '__main__':
    unittest.main()