# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import argparse
import os
import subprocess
import yapf

FLAGS = None
FORMAT_EXTS = ('proto', 'cc', 'cu', 'h')
SKIP_PATHS = ('tools',)


def visit(path):
    if FLAGS.verbose:
        print("visiting " + path)

    valid_ext = False
    python_file = False
    for ext in FORMAT_EXTS:
        if path.endswith('.' + ext):
            valid_ext = True
            break
    if path.endswith('.py'):
        valid_ext = True
        python_file = True
    if not valid_ext:
        if FLAGS.verbose:
            print("skipping due to extension: " + path)
        return True

    for skip in SKIP_PATHS:
        if path.startswith(skip):
            if FLAGS.verbose:
                print("skipping due to path prefix: " + path)
            return True
    if python_file:
        yapf.yapflib.yapf_api.FormatFile(path,
                                         in_place=True,
                                         style_config='google')
        return True
    else:
        args = ['clang-format-6.0', '--style=file', '-i']
        if FLAGS.verbose:
            args.append('-verbose')
        args.append(path)

        ret = subprocess.call(args)
        if ret != 0:
            print("format failed for " + path)
            return False

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('paths',
                        type=str,
                        nargs='*',
                        default=None,
                        help='Directories or files to format')
    FLAGS = parser.parse_args()

    # Check the version of yapf. Needs a consistent version
    # of yapf to prevent unneccessary changes in the code.
    if (yapf.__version__ != '0.30.0'):
        print("Needs yapf 0.30.0, but got yapf {}".format(yapf.__version__))

    if (FLAGS.paths is None) or (len(FLAGS.paths) == 0):
        parser.print_help()
        exit(1)

    ret = True
    for path in FLAGS.paths:
        if not os.path.isdir(path):
            if not visit(path):
                ret = False
        else:
            for root, dirs, files in os.walk(path):
                for name in files:
                    if not visit(os.path.join(root, name)):
                        ret = False

    exit(0 if ret else 1)
