set(COMMON_SOURCES c/mnist_util.c)

add_executable(mnist_model_size c/model_size.c ${COMMON_SOURCES})
target_include_directories(mnist_model_size PRIVATE c)
target_link_libraries(mnist_model_size PRIVATE tsetlin_machine_c)

add_executable(mnist_pretrained_inference c/pretrained_inference_demo.c ${COMMON_SOURCES})
target_include_directories(mnist_pretrained_inference PRIVATE c)
target_link_libraries(mnist_pretrained_inference PRIVATE tsetlin_machine_c)

add_executable(mnist_training c/training_demo.c ${COMMON_SOURCES})
target_include_directories(mnist_training PRIVATE c)
target_link_libraries(mnist_training PRIVATE tsetlin_machine_c)
