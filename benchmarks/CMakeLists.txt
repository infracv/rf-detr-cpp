# benchmarks/CMakeLists.txt — included from root when RFDETR_BUILD_BENCHMARKS=ON.
#
# rfdetr_bench itself is built in the root CMakeLists under RFDETR_BUILD_APPS.
# This subdirectory only ensures the results/ directory exists at build time
# and can be extended with additional benchmark targets (e.g. accuracy scripts).

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/results)
