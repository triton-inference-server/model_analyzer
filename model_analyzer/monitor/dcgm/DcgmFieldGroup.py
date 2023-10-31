# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import model_analyzer.monitor.dcgm.dcgm_agent as dcgm_agent
import model_analyzer.monitor.dcgm.dcgm_structs as dcgm_structs
'''
Class for managing a group of field IDs in the host engine.
'''


class DcgmFieldGroup:
    '''
    Constructor

    dcgmHandle - DcgmHandle() instance to use for communicating with the host engine
    name - Name of the field group to use within DCGM. This must be unique
    fieldIds - Fields that are part of this group
    fieldGroupId - If provided, this is used to initialize the object from an existing field group ID
    '''

    def __init__(self, dcgmHandle, name="", fieldIds=None, fieldGroupId=None):
        fieldIds = fieldIds or []
        self.name = name
        self.fieldIds = fieldIds
        self._dcgmHandle = dcgmHandle
        self.wasCreated = False

        #If the user passed in an ID, the field group already exists. Fetch live info
        if fieldGroupId is not None:
            self.fieldGroupId = fieldGroupId
            fieldGroupInfo = dcgm_agent.dcgmFieldGroupGetInfo(
                self._dcgmHandle.handle, self.fieldGroupId)
            self.name = fieldGroupInfo.fieldGroupName
            self.fieldIds = fieldGroupInfo.fieldIds
        else:
            self.fieldGroupId = None  #Assign here so the destructor doesn't fail if the call below fails
            self.fieldGroupId = dcgm_agent.dcgmFieldGroupCreate(
                self._dcgmHandle.handle, fieldIds, name)
            self.wasCreated = True

    '''
    Remove this field group from DCGM. This object can no longer be passed to other APIs after this call.
    '''

    def Delete(self):
        if self.wasCreated and self.fieldGroupId is not None:
            try:
                try:
                    dcgm_agent.dcgmFieldGroupDestroy(self._dcgmHandle.handle,
                                                     self.fieldGroupId)
                except dcgm_structs.dcgmExceptionClass(
                        dcgm_structs.DCGM_ST_NO_DATA):
                    # someone may have deleted the group under us. That's ok.
                    pass
                except dcgm_structs.dcgmExceptionClass(
                        dcgm_structs.DCGM_ST_CONNECTION_NOT_VALID):
                    # We lost our connection, but we're destructing this object anyway.
                    pass
            except AttributeError as ae:
                # When we're cleaning up at the end, dcgm_agent and dcgm_structs have been unloaded and we'll
                # get an AttributeError: "'NoneType' object has no 'dcgmExceptionClass'" Ignore this
                pass
            except TypeError as te:
                # When we're cleaning up at the end, dcgm_agent and dcgm_structs have been unloaded and we might
                # get a TypeError: "'NoneType' object is not callable'" Ignore this
                pass
            self.fieldGroupId = None
            self._dcgmHandle = None

    #Destructor
    def __del__(self):
        self.Delete()
