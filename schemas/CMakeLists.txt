set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

set(TM_SCHEMA_FILE ${CMAKE_CURRENT_SOURCE_DIR}/tsetlin_machine.fbs)
set(TM_GENERATED_HEADERS
    ${GENERATED_DIR}/flatbuffers_common_builder.h
    ${GENERATED_DIR}/flatbuffers_common_reader.h
    ${GENERATED_DIR}/tsetlin_machine_builder.h
    ${GENERATED_DIR}/tsetlin_machine_reader.h
    ${GENERATED_DIR}/tsetlin_machine_verifier.h
)

add_custom_command(
    OUTPUT ${TM_GENERATED_HEADERS}
    COMMAND flatcc_cli -a -o ${GENERATED_DIR} ${TM_SCHEMA_FILE}
    DEPENDS ${TM_SCHEMA_FILE} flatcc_cli
    COMMENT "Generating flatbuffers code from ${TM_SCHEMA_FILE}"
)

add_custom_target(
    tsetlin_machine_fbs_codegen
    DEPENDS ${TM_GENERATED_HEADERS}
)

add_library(tsetlin_machine_fbs INTERFACE)
add_dependencies(tsetlin_machine_fbs tsetlin_machine_fbs_codegen)

target_include_directories(tsetlin_machine_fbs
    INTERFACE ${GENERATED_DIR}
)

target_link_libraries(tsetlin_machine_fbs
    INTERFACE flatccrt
)
