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
    # provide generated schema include paths and flatcc headers only while building
    target_include_directories(tsetlin_machine_c
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/schemas/generated>
            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/schemas/generated>
    )
    target_link_libraries(tsetlin_machine_c
        PRIVATE $<BUILD_INTERFACE:tsetlin_machine_fbs>
        PUBLIC $<BUILD_INTERFACE:flatccrt>
    )
    # install(TARGETS flatccrt
    #     COMPONENT tsetlin_machine_c
    #     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    #     ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    #     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    # )
endif()

# rm -rf build install \
# && cmake -B build -DCMAKE_INSTALL_PREFIX=install \
# && cmake --build build \
# && cmake --install build --component tsetlin_machine_c
include(GNUInstallDirs)

install(TARGETS tsetlin_machine_c
    EXPORT tsetlin_machine_c-targets
    COMPONENT tsetlin_machine_c
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FILE_SET HEADERS DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT tsetlin_machine_c-targets
    FILE tsetlin_machine_c-config.cmake
    NAMESPACE tsetlin_machine_c::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/tsetlin_machine_c
    COMPONENT tsetlin_machine_c
)
