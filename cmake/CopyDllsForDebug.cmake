# This is a helper script to run BundleUtilities fixup_bundle as postbuild
# for a target. The primary use case is to copy .DLLs to the build directory for
# the Windows platform. It allows generator expressions to be used to determine
# the binary location
#
# Usage : copy_dlls_for_debug TARGET LIBS DIRS
# - TARGET : A cmake target
# - See fixup_bundle for LIBS and DIRS arguments

# MIT License

# Copyright (c) 2018 Jakob Weiss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


if(RUN_IT)
# Script ran by the add_custom_command
	include(BundleUtilities)
	fixup_bundle("${TO_FIXUP_FILE}" "${TO_FIXUP_LIBS}" "${TO_FIXUP_DIRS}")
# End of script ran by the add_custom_command
else()

set(THIS_FILE ${CMAKE_CURRENT_LIST_FILE})
function(copy_dlls_for_debug _target _libs _dirs)
	if(WIN32)
		add_custom_command(
			TARGET ${_target} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -DRUN_IT:BOOL=ON -DTO_FIXUP_FILE=$<TARGET_FILE:${_target}> -DTO_FIXUP_LIBS=${_libs} -DTO_FIXUP_DIRS=${_dirs} -P ${THIS_FILE}
			COMMENT "Fixing up dependencies for ${_target}"
			VERBATIM
		)
	endif(WIN32)
endfunction()

endif()

