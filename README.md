# And-Or Graph(AOG) Library
This is a C++ implementation of a general purpose AOG library.

## Requirements
* C++ Boost Libraries
* CMake is useful for building

## Instructions
To use this library, first install C++ boost libraries. You can download it from <http://www.boost.org/>, or by your system's package manager. For example, if you are using macOS, run the following command in your terminal:  
`brew install boost`

For CMake, you can download it from <https://cmake.org/download/>, or by your system's package manager, for example:  
`brew install cmake`

Then git clone this repository:  
`git clone https://github.com/vcla/STAT232B_AOGLib.git`

And change to the directory:  
`cd STAT232B_AOGLib/`

Now create a directory for running CMake:  
`mkdir cmake-build-debug`

And change to the newly created folder:  
`cd cmake-build-debug/`

After that, in the "cmake-build-debug" folder, run:  
`cmake ../`

And run:  
`make`

to compile the library. Finally run:  
`./AOG_Base/graph`

to start the main sample program.

## AOG_LIB :: AOG
### Constructors
Constructors | Description
------------|--------------
`AOG()` | Default Constructor
`AOG(Symbolic_Rule<StateType> rules)` | Construct a AOG from a set of symbolic rules
`AOG(Symbolic_State<StateType> leaf_states)` | Construct a AOG from a set of leaf symbolic states. No edge will be added to the AOG
`AOG(const AOG& AOG_)` | Copy constructor
`AOG(const string& AOG_)` | Reconstrut an AOG from a Saved file, assume templated type have constructor using string argument
  
### Methods
Methods | Description
--------|--------------
`unsigned NumOfRules()` | Return the number of rules contained in the AOG.
`unsigned NumOfStates()` | Return the number of symbolic states in the AOG.
`unsigned NumofLeafStates()` | Return the number of leaf symbolic states in AOG.
`bool ExistRule(Symbolic_Rule<StateType> rule)` | Return true if the given rule exists in the AOG. Note that A->BC and A->CB are considered different rules
`vector<Symbolic_Rule <StateType> > GetRules()` | Return all the rules contained in the AOG.
`vector<Symbolic_State <StateType> > GetStates()` | Return all symbolic states in the AOG.
`vector<Symbolic_State <StateType> > GetLeafStates()` | Return all leaf symbolic states in the AOG.
`VertexId GetVertexIdByState(Symbolic_State<StateType> state)` | Return the vertex id of the given state.
`Symbolic_State<StateType> GetStateByVertexId(VertexId source)` | Return the symbolic state that has the given vertex id
`VertexId AddVertex(shared_ptr<AOG_Vertex<StateType> > aog_vertex)` | Add the given vertex into AOG and return the vertex id assigned to this vertex.
`bool AddEdge(VertexId source, VertexId target, shared_ptr<AOG_Edge> aog_edge,bool multi_edge)`| parameters:<br/> &nbsp;&nbsp;&nbsp;&nbsp;source: the vertex id of the source of the edge<br/>&nbsp;&nbsp;&nbsp;&nbsp;target: the vertex id of the target of the edge<br/>&nbsp;&nbsp;&nbsp;&nbsp; aog_edge: an AOG_Edge object that contains the weight of the edge to be added
`bool DeleteVertex(VertexId vid)` | Return true if the vertex indicated by the given vertex id is successfully deleted
`vector<Symbolic_Rule<StateType> > GetRulesAsSource(const Symbolic_State<StateType> &) const` | Get all rules with the query state as the source
`vector<Symbolic_Rule<StateType> > GetRulesAsTarget(const Symbolic_State<StateType> &) const` | Get all rules with the query state as one of the targets.
`void SetIsAnd(VertexId source_id, bool is_and)` | Set a vertex to an And-node or an Or-node
`void SetVertexAttribute(VertexId source_id, const AttributeType& new_attribute)` | Set a vertex's attribute
`void SetVertexPotential(VertexId, const double)` | Set a vertex's potential, can be used for terminal nodes
`void SetVertexPotentialFunc(VertexId source_id, const std::function<double(AOG_Vertex<StateType, AttributeType>&,const std::vector<std::shared_ptr<AOG_Vertex<StateType, AttributeType> > >&)>)` | Set a vertex's potential function
`void UpdateVertexPotential(VertexId v_id)` | Let a vertex calculate its potential based on its neighbors
`void AddVertexNeighbor(VertexId self_id, VertexId neighbor_id, bool check_dup = true)` | Add a neighbor to a vertex, neighbors are for potential calculation only
`bool DeleteVertexNeighbor(VertexId self_id, VertexId neighbor_id)` | Delete a neighbor of a vertex
`bool SetOutEdgeWeights(vertexId source, unordered_map<VertexId, double> weights)` | Reset the weights of all the outedges of a given Or-node. Return true if weights are set successfully
`unordered_map<VertexId, double> GetOutEdgeWeights(VertexId source, bool is_normalized)` | return an unordered_map that maps all the ids of outedges' target vertices to their corrensponding weights. The weights are normalized if is_normalized = true
`unordered_map<VertexId,double> Normalize(VertexId src_id)` | Normalize the branching probabilities of the given Or-node, and return the normalized weights.
`unordered_map<VertexId, vector<VertexId>> Sample(VertexId root, vector<VertexId>&, double&)` | Sample a parse graph of the AOG starting from the given root vertex, parse graph is returned while terminal nodes and probability are updated by passing by reference. 
`void AddRule(Symbolic_Rule<StateType> rule)` | Add a rule to the AOG and update it.
`bool DeleteRule(Symbolic_Rule<StateType> rule)` | Delete a rule in AOG and update it. Return false if the given rule does not exist.
`vector<Symbolic_State<StateType> > GetTopLevelStates()` | Return states that aren't targets of any rules. Normally this should only be the root.
`void DeleteNoParentRules(const Symbolic_State<StateType>& state)` | Remove rules whose source node has no parent, query by a state. Root node won't be touched by this method. 
`void TruncateGraph()` | Remove unnecessary dummy nodes in current AOG, should be called before visualization or saving.
`void Visualize(string, string)` | Visualize the AOG. See [External](./External) folder for detailed usage.
`void SaveGraph(string path, string filename)` | Save current AOG into a file ready for constructor to read in, assume StateType support fstream print

