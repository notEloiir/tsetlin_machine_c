add_executable(test_runner)

target_sources(test_runner 
    PRIVATE
        c/test_runner.c
        c/test_linked_list.c
        c/test_tsetlin_machine.c
)

# target_include_directories(test_runner 
#     PRIVATE
#         ${CMAKE_SOURCE_DIR}/src/tsetlin_machine_c/include
# )
# add_dependencies(test_runner tsetlin_machine_fbs_codegen)

target_link_libraries(test_runner 
    PRIVATE
        unity::framework
        tsetlin_machine_c  # possible duplicate symbols
)

add_test(NAME test_runner COMMAND test_runner)
