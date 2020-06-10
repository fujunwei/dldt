# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME inference_engine_builder_api)
#set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADERS ${InferenceEngine_C_API_SOURCE_DIR}/include/*.h)

function(nn_builder_disable_warnings)
    disable_deprecated_warnings()
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR} PROPERTIES COMPILE_FLAGS "${ie_c_cxx_deprecated}")
endfunction()

nn_builder_disable_warnings()

# create library

add_library(${TARGET_NAME} SHARED ${HEADERS} ${SOURCES})

target_link_libraries(${TARGET_NAME} PRIVATE inference_engine inference_engine_nn_builder)

target_include_directories(${TARGET_NAME} PUBLIC "${InferenceEngine_C_API_SOURCE_DIR}/include")

add_cpplint_target(${TARGET_NAME}_cpplint FOR_TARGETS ${TARGET_NAME})

# export

export(TARGETS ${TARGET_NAME} NAMESPACE IE:: APPEND FILE "${CMAKE_BINARY_DIR}/targets.cmake")

# install

install(TARGETS ${TARGET_NAME}
        RUNTIME DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core
        ARCHIVE DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core
        LIBRARY DESTINATION ${IE_CPACK_LIBRARY_PATH} COMPONENT core)

install(DIRECTORY ${InferenceEngine_C_API_SOURCE_DIR}/include/
        DESTINATION ${IE_CPACK_IE_DIR}/include/
        COMPONENT core)
