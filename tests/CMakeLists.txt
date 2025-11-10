add_executable(test_runner c/test_runner.c)
target_link_libraries(test_runner PRIVATE tsetlin_machine_c Unity)
add_test(NAME test_runner COMMAND test_runner)