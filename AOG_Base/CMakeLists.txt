set(GRAPH_SOURCE_FILES
        ../Core/Graph.hpp
        AOG_Edge.h
        AOG_Vertex.h
        Symbolic_Rule.h
        AOG.h
        Symbolic_State.h)

set(GRAPH_TEST_FILES
        grammar_example.cpp)

add_executable(graph ${GRAPH_SOURCE_FILES} ${GRAPH_TEST_FILES})

target_link_libraries(graph ${Boost_LIBRARIES})