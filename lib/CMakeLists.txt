cmake_minimum_required (VERSION 2.8.8)

project(pyblis)

option(PYBLIS_BUNDLE_BLIS
       "Vendor BLIS library as part of package"
       ON)

option(PYBLIS_BUILD_BLIS
       "Link against BLIS built by this project. If not set, will look for a local version."
       OFF)

include(ExternalProject)

if(PYBLIS_BUNDLE_BLIS)
    set(LIBBLIS_NAME "libblis.a")
    add_library(blis STATIC IMPORTED)
else()
    set(LIBBLIS_NAME "blis")
    add_library(blis SHARED IMPORTED)
endif()

if(PYBLIS_BUILD_BLIS)
    message(STATUS "Building BLIS from source")
    set(BLIS_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/blis_ep/src/blis_ep-install")
    set(BLIS_INCLUDE_DIR "${BLIS_PREFIX}/include")
    set(BLIS_LIB "${BLIS_PREFIX}/lib/${LIBBLIS_NAME}")
    if(APPLE)
        set(BLIS_CONFIGURE
            "./configure"
            "--prefix=${BLIS_PREFIX}"
            "--disable-blas"
            "--enable-threading=pthreads"
            "intel64")
    else()
        set(BLIS_CONFIGURE
            "./configure"
            "--prefix=${BLIS_PREFIX}"
            "--disable-blas"
            "--enable-threading=pthreads"
            "x86_64")
    endif()
    ExternalProject_Add(blis_ep
                        INSTALL_DIR ${BLIS_PREFIX}
                        URL https://github.com/flame/blis/archive/0.6.0.tar.gz
                        BUILD_IN_SOURCE 1
                        CONFIGURE_COMMAND ${BLIS_CONFIGURE}
                        BUILD_COMMAND ${MAKE} ${MAKE_BUILD_ARGS}
                        BUILD_BYPRODUCTS ${BLIS_LIB})
    file(MAKE_DIRECTORY "${BLIS_PREFIX}/include")
    add_dependencies(blis blis_ep)
else()
    # Look for local blis library
    find_library(BLIS_LIB
                 NAMES "${LIBBLIS_NAME}"
                 PATH_SUFFIXES "${LIB_PATH_SUFFIXES}")
    find_path(BLIS_INCLUDE_DIR
              NAMES blis/blis.h
              PATH_SUFFIXES ${INCLUDE_PATH_SUFFIXES})

    if(NOT BLIS_LIB OR NOT BLIS_INCLUDE_DIR)
        message(FATAL_ERROR "Unable to find libblis, please install or set PYBLIS_BUILD_BLIS")
    endif()
endif()

set_target_properties(
    blis
    PROPERTIES IMPORTED_LOCATION "${BLIS_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${BLIS_INCLUDE_DIR}"
)

set(SOURCE_FILES
    src/pyblis.c)

add_library(pyblis SHARED ${SOURCE_FILES})
target_link_libraries(pyblis blis)
target_include_directories(pyblis INTERFACE blis)
if(APPLE)
    set_target_properties(pyblis PROPERTIES LINK_FLAGS "-Wl,-exported_symbols_list,\"${CMAKE_CURRENT_SOURCE_DIR}/pyblis.syms\"")
else()
    set_target_properties(pyblis PROPERTIES LINK_FLAGS "-Wl,-version-script,\"${CMAKE_CURRENT_SOURCE_DIR}/pyblis.syms\"")
endif()
