import donaldson
import numpy as np
import unittest

class DonaldsonTests(unittest.TestCase):
    def test_to_good_coordinates_point_in_affine_expect_exclude_first_coordinate(self):
        p_affine = np.array([ 1.+0.j,  0.5+0.5j, -0.3-0.3j,  0.+0.j, 0.-0.j])
        expected_good_coord = lambda x : ((-1) - x[2]**5 - x[3]**5 - x[4] **5) ** (1/5)

        res = donaldson.to_good_coordinates(p_affine)

        self.assertEqual(res[0], 1+0.j)
        self.assertEqual(res[1], expected_good_coord(p_affine))

    def test_find_max_dq_coord_index_p_affine_expect_second_coord(self):
        """test find_max_dq_coord_index should ignore the first coordinate"""
        p_affine = np.array([ 1.+0.j,  0.5+0.5j, -0.3-0.3j,  0.+0.j, 0.-0.j])

        index = donaldson.find_max_dq_coord_index(p_affine)

        self.assertEqual(index, 1)

if __name__ == '__main__':
    unittest.main()