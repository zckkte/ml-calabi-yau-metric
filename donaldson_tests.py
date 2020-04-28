import donaldson
import numpy as np
import unittest

class DonaldsonTests(unittest.TestCase):
    def test_find_max_dq_coord_index_p_affine_expect_second_coord(self):
        """test find_max_dq_coord_index should ignore the first coordinate"""
        p_affine = np.array([ 1.+0.j,  0.5+0.5j, -0.3-0.3j,  0.+0.j, 0.-0.j])

        index = donaldson.find_max_dq_coord_index(p_affine)

        self.assertEqual(index, 1)

    def test_vectorize_weight_10_points_expect_10_weights(self):
        """test vectorize weight should return 10 weights given input of 10 points"""
        n_random_points = lambda n : np.random.rand(n*5, 2).astype(float).view(np.complex128).reshape(n,5)
        points = n_random_points(10)

        vweight = np.vectorize(donaldson.weight, signature="(m)->()")
        weights = vweight(points)

        self.assertEqual(len(weights), 10)

if __name__ == '__main__':
    unittest.main()