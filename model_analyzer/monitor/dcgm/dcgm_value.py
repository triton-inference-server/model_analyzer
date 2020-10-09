# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

DCGM_INT32_BLANK = 0x7ffffff0
DCGM_INT64_BLANK = 0x7ffffffffffffff0

# Base value for double blank. 2 ** 47. FP 64 has 52 bits of mantissa,
# so 47 bits can still increment by 1 and represent each value from 0-15
DCGM_FP64_BLANK = 140737488355328.0

DCGM_STR_BLANK = "<<<NULL>>>"

# Represents an error where data was not found
DCGM_INT32_NOT_FOUND = (DCGM_INT32_BLANK + 1)
DCGM_INT64_NOT_FOUND = (DCGM_INT64_BLANK + 1)
DCGM_FP64_NOT_FOUND = (DCGM_FP64_BLANK + 1.0)
DCGM_STR_NOT_FOUND = "<<<NOT_FOUND>>>"

# Represents an error where fetching the value is not supported
DCGM_INT32_NOT_SUPPORTED = (DCGM_INT32_BLANK + 2)
DCGM_INT64_NOT_SUPPORTED = (DCGM_INT64_BLANK + 2)
DCGM_FP64_NOT_SUPPORTED = (DCGM_FP64_BLANK + 2.0)
DCGM_STR_NOT_SUPPORTED = "<<<NOT_SUPPORTED>>>"

# Represents and error where fetching the value is not allowed with our current
# credentials
DCGM_INT32_NOT_PERMISSIONED = (DCGM_INT32_BLANK + 3)
DCGM_INT64_NOT_PERMISSIONED = (DCGM_INT64_BLANK + 3)
DCGM_FP64_NOT_PERMISSIONED = (DCGM_FP64_BLANK + 3.0)
DCGM_STR_NOT_PERMISSIONED = "<<<NOT_PERM>>>"


###############################################################################
# Functions to check if a value is blank or not
def DCGM_INT32_IS_BLANK(val):
    if val >= DCGM_INT32_BLANK:
        return True
    else:
        return False


def DCGM_INT64_IS_BLANK(val):
    if val >= DCGM_INT64_BLANK:
        return True
    else:
        return False


def DCGM_FP64_IS_BLANK(val):
    if val >= DCGM_FP64_BLANK:
        return True
    else:
        return False


# Looks for <<< at first position and >>> inside string
def DCGM_STR_IS_BLANK(val):
    if 0 != val.find("<<<"):
        return False
    elif 0 > val.find(">>>"):
        return False
    return True


class DcgmValue:

    def __init__(self, value):
        # Contains either an integer (int64), string, or double of the actual
        # value
        self.value = value

    def SetFromInt32(self, i32Value):
        """
        Handle the special case where our source data was an int32 but is
        currently stored in a python int (int64), dealing with blanks
        """
        value = int(i32Value)

        if not DCGM_INT32_IS_BLANK(i32Value):
            self.value = value
            return

        if value == DCGM_INT32_NOT_FOUND:
            self.value = DCGM_INT64_NOT_FOUND
        elif value == DCGM_INT32_NOT_SUPPORTED:
            self.value = DCGM_INT64_NOT_SUPPORTED
        elif value == DCGM_INT32_NOT_PERMISSIONED:
            self.value = DCGM_INT64_NOT_PERMISSIONED
        else:
            self.value = DCGM_INT64_BLANK

    def IsBlank(self):
        """
        Returns True if the currently-stored value is a blank value. False if
        not
        """
        if self.value is None:
            return True
        elif type(self.value) == int or type(self.value) == long:
            return DCGM_INT64_IS_BLANK(self.value)
        elif type(self.value) == float:
            return DCGM_FP64_IS_BLANK(self.value)
        elif type(self.value) == str:
            return DCGM_STR_IS_BLANK(self.value)
        else:
            raise Exception("Unknown type: %s") % str(type(self.value))

    def __str__(self):
        return str(self.value)


def self_test():

    v = DcgmValue(1.0)
    assert (not v.IsBlank())
    assert (v.value == 1.0)

    v = DcgmValue(100)
    assert (not v.IsBlank())
    assert (v.value == 100)

    v = DcgmValue(DCGM_INT64_NOT_FOUND)
    assert (v.IsBlank())

    v = DcgmValue(DCGM_FP64_NOT_FOUND)
    assert (v.IsBlank())

    v.SetFromInt32(DCGM_INT32_NOT_SUPPORTED)
    assert (v.IsBlank())
    assert (v.value == DCGM_INT64_NOT_SUPPORTED)

    print("Tests passed")
    return


if __name__ == "__main__":
    self_test()
