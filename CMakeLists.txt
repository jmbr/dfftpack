cmake_minimum_required(VERSION 2.8)
project(dfftpack Fortran)
enable_language(Fortran)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

set(CMAKE_Fortran_FLAGS "-g")
set(CMAKE_Fortran_FLAGS_DEBUG "-O0")
set(CMAKE_Fortran_FLAGS_RELEASE "-O2 -funroll-loops -fexpensive-optimizations")

add_library(dfftpack zfftb.f cfftb1.f zfftf.f cfftf1.f zffti.f
cffti1.f dcosqb.f cosqb1.f dcosqf.f cosqf1.f dcosqi.f dcost.f dcosti.f
ezfft1.f dzfftb.f dzfftf.f dzffti.f passb.f passb2.f passb3.f passb4.f
passb5.f passf.f passf2.f passf3.f passf4.f passf5.f radb2.f radb3.f
radb4.f radb5.f radbg.f radf2.f radf3.f radf4.f radf5.f radfg.f
dfftb.f rfftb1.f dfftf.f rfftf1.f dffti.f rffti1.f dsinqb.f dsinqf.f
dsinqi.f dsint.f sint1.f dsinti.f)

enable_testing()
add_test(test-dfftpack test-dfftpack)

add_executable(test-dfftpack test.f)
target_link_libraries(test-dfftpack dfftpack)
