# Note: CMake support is community-based. The maintainers do not use CMake
# internally.

cmake_minimum_required(VERSION 2.8.8)

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(googletest-distribution)
set(GOOGLETEST_VERSION 1.10.0)

# This seems to fix errors like these:
#gtestd.lib(gtest-all.cc.obj) : error LNK2038: mismatch detected for 'RuntimeLibrary': value 'MTd_StaticDebug' doesn't match value 'MDd_DynamicDebug' in test_sift.cpp.obj
#msvcprtd.lib(MSVCP140D.dll) : error LNK2005: "public: struct _Cvtvec __cdecl std::_Locinfo::_Getcvt(void)const " (?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ) already defined in gtestd.lib(gtest-all.cc.obj)
#msvcprtd.lib(MSVCP140D.dll) : error LNK2005: "public: bool __cdecl std::ios_base::good(void)const " (?good@ios_base@std@@QEBA_NXZ) already defined in gtestd.lib(gtest-all.cc.obj)
#msvcprtd.lib(MSVCP140D.dll) : error LNK2005: "public: int __cdecl std::ios_base::flags(void)const " (?flags@ios_base@std@@QEBAHXZ) already defined in gtestd.lib(gtest-all.cc.obj)
#...
#libcpmtd.lib(locale0.obj) : error LNK2038: mismatch detected for 'RuntimeLibrary': value 'MTd_StaticDebug' doesn't match value 'MDd_DynamicDebug' in test_sift.cpp.obj
#libcpmtd.lib(locale0.obj) : error LNK2005: "void __cdecl std::_Facet_Register(class std::_Facet_base *)" (?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z) already defined in msvcprtd.lib(locale0_implib.obj)
#libcpmtd.lib(locale0.obj) : error LNK2005: "private: static class std::locale::_Locimp * __cdecl std::locale::_Getgloballocale(void)" (?_Getgloballocale@locale@std@@CAPEAV_Locimp@12@XZ) already defined in msvcprtd.lib(MSVCP140D.dll)
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# See https://github.com/MarinaKalashina/DateConverter_withTests/blob/351ca2841597189487d3c6a8e59fede4ef156aa1/Google_tests/lib/CMakeLists.txt#L15
if (CMAKE_VERSION VERSION_GREATER "3.0.2")
  if(NOT CYGWIN AND NOT MSYS AND NOT ${CMAKE_SYSTEM_NAME} STREQUAL QNX)
    set(CMAKE_CXX_EXTENSIONS OFF)
  endif()
endif()

enable_testing()

include(CMakeDependentOption)
include(GNUInstallDirs)

add_subdirectory( googletest )
