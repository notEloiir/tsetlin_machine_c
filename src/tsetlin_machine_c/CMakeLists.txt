add_library(tsetlin_machine_c)

target_sources(tsetlin_machine_c
    PRIVATE
        src/tsetlin_machine.c
        src/fast_prng.c
        src/sparse_tsetlin_machine.c
        src/stateless_tsetlin_machine.c

    PUBLIC
        FILE_SET HEADERS
        BASE_DIRS
            include
        FILES
            include/tsetlin_machine.h
            include/fast_prng.h
            include/sparse_tsetlin_machine.h
            include/stateless_tsetlin_machine.h
)

if(BUILD_FLATCC)
    add_dependencies(tsetlin_machine_c tsetlin_machine_fbs_codegen)
    target_link_libraries(tsetlin_machine_c PUBLIC tsetlin_machine_fbs)
endif()
