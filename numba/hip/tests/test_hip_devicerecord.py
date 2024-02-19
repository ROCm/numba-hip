# Copyright (c) 2012, Anaconda, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# MIT License
#
# Modifications Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import ctypes
from numba.hip.hipdrv.devicearray import (DeviceRecord, from_record_like,
                                          auto_device)
from numba.hip.testing import unittest, HIPTestCase as CUDATestCase
# from numba.hip.testing import skip_on_cudasim
from numba.np import numpy_support
from numba import hip as cuda

N_CHARS = 5


# equivalent to
# typedef struct {
#   double a; // 8 
#   int b;    // 4
#   double c[2]; // 2x8
#   char d[5]; // 5
# } mystruct; // unaligned size: 8 + 4 + 2x8 + 5 = 33
#             // aligned size: 8 + (4+4) + 2x8 + (5+3) = 5x8 = 40
recordtype = np.dtype(
    [
        ('a', np.float64),
        ('b', np.int32),
        ('c', np.complex64),
        ('d', (np.str_, N_CHARS))
    ],
    align=True
)

recordwitharray = np.dtype(
    [
        ('g', np.int32),
        ('h', np.float32, 2)
    ],
    align=True
)

recwithmat = np.dtype([('i', np.int32),
                       ('j', np.float32, (3, 3))])

recwithrecwithmat = np.dtype([('x', np.int32), ('y', recwithmat)])


# @skip_on_cudasim('Device Record API unsupported in the simulator')
class TestCudaDeviceRecord(CUDATestCase):
    """
    Tests the DeviceRecord class with np.void host types.
    """
    def setUp(self):
        super().setUp()
        self._create_data(np.zeros)

    def _create_data(self, array_ctor):
        self.dtype = np.dtype([('a', np.int32), ('b', np.float32)], align=True)
        self.hostz = array_ctor(1, self.dtype)[0]
        self.hostnz = array_ctor(1, self.dtype)[0]
        self.hostnz['a'] = 10
        self.hostnz['b'] = 11.0

    def _check_device_record(self, reference, rec):
        self.assertEqual(rec.shape, tuple())
        self.assertEqual(rec.strides, tuple())
        self.assertEqual(rec.dtype, reference.dtype)
        self.assertEqual(rec.alloc_size, reference.dtype.itemsize)
        self.assertIsNotNone(rec.gpu_data)
        self.assertNotEqual(rec.device_ctypes_pointer, ctypes.c_void_p(0))

        numba_type = numpy_support.from_dtype(reference.dtype)
        self.assertEqual(rec._numba_type_, numba_type)

    def test_device_record_interface(self):
        hostrec = self.hostz.copy()
        devrec = DeviceRecord(self.dtype)
        self._check_device_record(hostrec, devrec)

    def test_device_record_copy(self):
        hostrec = self.hostz.copy()
        devrec = DeviceRecord(self.dtype)
        devrec.copy_to_device(hostrec)

        # Copy back and check values are all zeros
        hostrec2 = self.hostnz.copy()
        devrec.copy_to_host(hostrec2)
        np.testing.assert_equal(self.hostz, hostrec2)

        # Copy non-zero values to GPU and back and check values
        hostrec3 = self.hostnz.copy()
        devrec.copy_to_device(hostrec3)

        hostrec4 = self.hostz.copy()
        devrec.copy_to_host(hostrec4)
        np.testing.assert_equal(hostrec4, self.hostnz)

    def test_from_record_like(self):
        # Create record from host record
        hostrec = self.hostz.copy()
        devrec = from_record_like(hostrec)
        self._check_device_record(hostrec, devrec)

        # Create record from device record and check for distinct data
        devrec2 = from_record_like(devrec)
        self._check_device_record(devrec, devrec2)
        self.assertNotEqual(devrec.gpu_data, devrec2.gpu_data)

    def test_auto_device(self):
        # Create record from host record
        hostrec = self.hostnz.copy()
        devrec, new_gpu_obj = auto_device(hostrec)
        self._check_device_record(hostrec, devrec)
        self.assertTrue(new_gpu_obj)

        # Copy data back and check it is equal to auto_device arg
        hostrec2 = self.hostz.copy()
        devrec.copy_to_host(hostrec2)
        np.testing.assert_equal(hostrec2, hostrec)


class TestCudaDeviceRecordWithRecord(TestCudaDeviceRecord):
    """
    Tests the DeviceRecord class with np.record host types
    """
    def setUp(self):
        CUDATestCase.setUp(self)
        self._create_data(np.recarray)


# @skip_on_cudasim('Structured array attr access not supported in simulator')
class TestRecordDtypeWithStructArrays(CUDATestCase):
    '''
    Test operation of device arrays on structured arrays.
    '''

    def _createSampleArrays(self):
        self.sample1d = cuda.device_array(3, dtype=recordtype)
        self.samplerec1darr = cuda.device_array(1, dtype=recordwitharray)[0]
        self.samplerecmat = cuda.device_array(1,dtype=recwithmat)[0]

    def setUp(self):
        super().setUp()
        self._createSampleArrays()

        ary = self.sample1d
        # print(ary)
        # print(ary.size)
        for i in range(ary.size):
            x = i + 1
            ary[i]['a'] = x / 2
            ary[i]['b'] = x
            ary[i]['c'] = x * 1j
            ary[i]['d'] = str(x) * N_CHARS

    def test_structured_array1(self):
        ary = self.sample1d
        for i in range(self.sample1d.size):
            x = i + 1
            self.assertEqual(ary[i]['a'], x / 2)
            self.assertEqual(ary[i]['b'], x)
            self.assertEqual(ary[i]['c'], x * 1j)
            self.assertEqual(ary[i]['d'], str(x) * N_CHARS)

    # TODO HIP currently expected to fail
    def test_structured_array2(self):
        """NOTE: Requires JIT capabilities.
        """
        ary = self.samplerec1darr
        ary['g'] = 2
        ary['h'][0] = 3.0
        ary['h'][1] = 4.0
        self.assertEqual(ary['g'], 2)
        self.assertEqual(ary['h'][0], 3.0)
        self.assertEqual(ary['h'][1], 4.0)

    # TODO HIP currently expected to fail
    def test_structured_array3(self):
        """NOTE: Requires JIT capabilities.
        """
        ary = self.samplerecmat
        mat = np.array([[5.0, 10.0, 15.0],
                       [20.0, 25.0, 30.0],
                       [35.0, 40.0, 45.0]],
                       dtype=np.float32).reshape(3,3)
        ary['j'][:] = mat
        np.testing.assert_equal(ary['j'], mat)

    # TODO HIP currently expected to fail
    def test_structured_array4(self):
        """NOTE: Requires JIT capabilities.
        """
        arr = np.zeros(1, dtype=recwithrecwithmat)
        d_arr = cuda.to_device(arr)
        d_arr[0]['y']['i'] = 1
        self.assertEqual(d_arr[0]['y']['i'], 1)
        d_arr[0]['y']['j'][0, 0] = 2.0
        self.assertEqual(d_arr[0]['y']['j'][0, 0], 2.0)


if __name__ == '__main__':
    unittest.main()
