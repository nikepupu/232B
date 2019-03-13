//
// Created by yuanluyao on 10/23/17.
//

#ifndef AOG_LIB_AOG_H
#define AOG_LIB_AOG_H

#include <memory>
#include <random>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <stack>
#include <utility>
#include <algorithm>

#include "../Core/Graph.hpp"
#include "AOG_Vertex.h"
#include "AOG_Edge.h"
#include "Symbolic_Rule.h"

template <class StateType, class AttributeType>
using AOG_Graph = AOG_LIB::Graph<AOG_LIB::AOG_Vertex<StateType, AttributeType>, AOG_LIB::AOG_Edge>;

namespace AOG_LIB
{
    /* This class defines an AOG graph with basic operations
     * An AOG graph consists of And-nodes and Or-nodes with 
     * their own semantic meanings and edges with different weights.
     * Each vertex in the AOG graph has its own unique id.
     */
    template<class StateType, class AttributeType>
    class AOG : public AOG_Graph<StateType, AttributeType>
    {
        // all the rules stored in a hash table for easy look up
        std::unordered_set<Symbolic_Rule<StateType> > all_rules_;
        //all leaf states stored in a hash table for easy look up
        std::unordered_set<Symbolic_State<StateType> > all_leaf_states_;
        //a map from state to graph vertex id
        std::unordered_map<Symbolic_State<StateType>, VertexId> state_to_vertex_;
        // the vertex id of the root node
        VertexId root_id_;
        //an auxiliary boolean variable for error checking when adding a vertex
        bool has_root_;

    private:
        /* This function adds an edge between two vertices.      
         * Cannot add an edge that starts and ends at same vertex
         * AOG object should call AddRule instead of AddEdge
         * @params:
         *     source: the vertex id of the source node
         *     target: the vertex id of the target node
         *     aog_edge: the edge to be added from the source node to the target node
         *     multi_edge: true if the AOG graph allows multiple edges between two vertices
         * @return:
         *     true if the edge is added successfully
         */        
        virtual bool AddEdge(const VertexId, const VertexId,
                             const std::shared_ptr<AOG_Edge>,
                             bool = true, int = -1);

        /**
         * Disable DeleteEdge function for AOG class, use DeleteRule instead
        */
        using AOG_Graph<StateType, AttributeType>::DeleteEdge;

    public:
        //default constructor
        AOG();
    
        //construct from a set of rules. Currently it assigns the weight of 1.0 to all edges
        AOG(const std::vector<Symbolic_Rule<StateType> >&);

        //AOG copy constructor
        AOG(const AOG& other);

        //AOG constructor by reading in a file
        AOG(const std::string&);

        //construct from a set of leaf states
        explicit AOG(const std::vector<Symbolic_State<StateType> > &);
        //return the number of rules the AOG contains
        unsigned NumOfRules() const { return unsigned(this->all_rules_.size()); }
        //return the number of symbolic states in the AOG
        unsigned NumOfStates() const { return unsigned(this->state_to_vertex_.size()); }
        //return the number of leaf states in the AOG
        unsigned NumOfLeafStates() const { return unsigned(this->all_leaf_states_.size()); }
        //return the number of vertices in the AOG
        virtual unsigned NumberOfVertices() const {return state_to_vertex_.size();}
        //return whether the root of the graph is set
        bool HasRoot() const{return has_root_;}
        
        /** This function returns the size of the grammar count rules by rules
         * @param:
         *      leaf_bias: how much do leaf nodes count, can be smaller than non-terminal nodes
         *  @return:
         *      the grammar size of current graph
        */
        double GrammarSize(double leaf_bias);

        /* This function adds a vertex to the AOG graph
         * @param: 
         *      aog_vertex: the vertex to be added to the AOG graph
         * @throw: 
         *      throw an exception when user tries to add a second root node to the AOG graph
         * @return:
         *      an unsigned integer that indicates the id of the newly 
         *      added vertex(if the vertex alreadly exists, return the existing id).
         */    
        virtual VertexId AddVertex(const std::shared_ptr<AOG_Vertex<StateType, AttributeType> >);

        /* This function deletes a vertex and all related edges from the AOG graph
         * @params: 
         *     vid: the id of the vertex to be deleted
         * @return:
         *     true if the vertex is successfully
         */
        virtual bool DeleteVertex(const VertexId);

        /* This function returns all the rules used to construct the AOG graph
         * @param:
         *     void
         * @return:
         *     a vector that contains all the symbolic
         */
        std::vector<Symbolic_Rule<StateType> > GetRules() const;

        /* This function returns all the leaf states in the AOG graph
         * @param:
         *     void
         * @return:
         *     a vector that contains all the leaf states
         */
        std::vector<Symbolic_State<StateType> > GetLeafStates() const;

       /* This function returns all the symbolic states in the AOG graph
        * @param:
        *     void
        * @return:
        *     a vector that contains all the states in vertices in AOG graph
        */
        std::vector<Symbolic_State<StateType> > GetStates() const;

        /* @param: 
         *     source: the id of the vertex containing the desired state
         * @return:
         *     a Symbolic_State object that represents the state inside the given vertex
         */
        Symbolic_State<StateType> GetStateByVertexId(VertexId) const;

        /* @param:
         *     void
         * @return:
         *     the vertex id of the root vertex if there exist one, else throw an exception
         */
        VertexId GetRoot() const;

        /* @param:
         *     state: the symbolic state the user want to query about
         * @return:
         *     an unsigned integer that indicates the id of the vertex containing the given symbolic state
         */
        VertexId GetVertexIdByState(const Symbolic_State<StateType> &) const;

        /* This function gets weights of all the outedges from a given source vertex
         * @params:
         *      source: the id of the source vertex
         *      is_normalized: a boolean value. If true, return 
         *                     the normalized weights of all the outedges from the target vertex.
         * @return:
         *      return an unordered_map that maps all the outedges' 
         *      target vertices' ids to their corresponding weights
         */
        std::unordered_map<VertexId,double> GetOutEdgeWeights(VertexId, bool) const;

        /**
         * Returns all rules with a certain state as source
         * @params:
         *      query state
         * @return:
         *      vector of rules with the query state as source, empty if no such rules
        */
        std::vector<Symbolic_Rule<StateType> > GetRulesAsSource(const Symbolic_State<StateType> &) const;

        /**
         * Returns all rules with a certain state as one of the targets
         * @params:
         *      query state
         * @return:
         *      vector of rules with the query state as one of the targets, empty if no such rules
        */
        std::vector<Symbolic_Rule<StateType> > GetRulesAsTarget(const Symbolic_State<StateType> &) const;

        /* This function changes a given node to an and/or-node
         * @params:
         *      source_id: the id of the vertex to be changed
         *      is_and: true if trying to set the vertex to an and-node
         * @return:
         *      void
         */
        void SetIsAnd(VertexId, const bool);

        /* This function changes a given vertex's attribute
         * @params:
         *      source_id: the id of the vertex to be changed
         *      new_attribute: The new attribute to be set
         * @return:
         *      void
         */
        void SetVertexAttribute(VertexId source_id, const AttributeType& new_attribute);

        /* This function changes a given node to an and/or-node
         * @params:
         *      source_id: the id of the vertex to be changed
         *      potential: the new potential of the vertex
         * @return:
         *      void
         */
        void SetVertexPotential(VertexId, const double);

        /* This function set the potential function for a given node
         * @params:
         *      source_id: the id of the vertex to be changed
         *      new_func: the potential function to be used
         * @return:
         *      void
         */
        void SetVertexPotentialFunc(VertexId source_id,
                                    const std::function<double(AOG_Vertex<StateType, AttributeType>&,
                                                               const std::vector<std::shared_ptr<AOG_Vertex<StateType, AttributeType> > >&)>);
 
        
        /* This function update the potential value for a given node
         * @params:
         *      source_id: the id of the vertex whose potential will be updated
         * @return:
         *      void
         */
        void UpdateVertexPotential(VertexId v_id);

        /* This function adds a neighbor to a vertex
         * @params:
         *      self_id: the id of the vertex whom the neighbor is added to
         *      neighbor_id: the id of the new neighbor
         *      check_dup: whether check neighbor duplicates
         * @return:
         *      void
         */
        void AddVertexNeighbor(VertexId self_id, VertexId neighbor_id, bool check_dup = true);

        /* This function removes a neighbor of a vertex
         * @params:
         *      self_id: the id of the vertex whom the neighbor is added to
         *      neighbor_id: the id of the neighbor to be removed
         * @return:
         *      bool: whether there was such a neighbor, false if the neighbor doesn't exist
         */
        bool DeleteVertexNeighbor(VertexId self_id, VertexId neighbor_id);

        /* This function reassigns weights to a given vertex's outedges
         * @params:
         *      source: the source vertex of all the outedges to be modified
         *      weight: an unordered_map that contains the mapping from the id 
         *              of outedges to the weights each outedge need to be set to
         * @return:
         *      true if the weights are set successfully
         */
        bool SetOutEdgeWeights(VertexId, const std::unordered_map<VertexId ,double>&);

        /* This function set the root of the AOG to the vertex holding content root
        * 
        */ 
        void SetRoot(Symbolic_State<StateType> root);

        /* This function normalizes all the weights from a given source vertex
         * @param:
         *      src_id: the id of the source vertex whose outedges will be normalized
         * @return:
         *      an unordered_map that maps ids of all target vertices of outedges to the normalized weights.
         */
        std::unordered_map<VertexId,double> Normalize(VertexId);

        /* This function checks a certain rule for its existence
         * A->BC and A->CB are considered different rules
         * @param:
         *      rule: the rule to be checked
         * @return:
         *      true if the rule already exists in AOG, false otherwise
         */
        bool ExistRule(const Symbolic_Rule<StateType>&);

        /* This function adds a rule to the AOG
         * @param:
         *      rule: the rule to be added
         * @return:
         *      bool: whether the add is successful
         */
        bool AddRule(const Symbolic_Rule<StateType>&);

        /* This function deletes corresponding edges and 
         * vertices in the AOG graph given the rule to be deleted
         * @param:
         *      rule: the rule to be deleted
         * @return:
         *      false if the given rule does not exist, true otherwise
         */
        bool DeleteRule(const Symbolic_Rule<StateType>&);

        /* This Sample() function eliminates all Or-nodes by sampling over edge weights
         * @param:
         *      root: The vertex to sample from
         * @return:
         *      a vector of vertex ID for the parse graph
         *      terminal leaf nodes and sample probability can be obtained with passing by reference
         */
        std::shared_ptr<std::vector<VertexId>> Sample(VertexId, std::vector<VertexId> &, double &) const;

        /* This GetTopLevelStates() function returns the top most level state(s) of the T-AOG 
         * @param:
         *      void
         * @return:
         *      a vector containing the top most level state(s) of the T-AOG 
         */
        std::vector<Symbolic_State<StateType> > GetTopLevelStates();
        
        /** This function can be used to remove rules whose source has no parent
         *  The delete is a recursive process
         *  @param:
         *      state: source state of the rule
         *  @return:
         *      void
         */ 
        void DeleteNoParentRules(const Symbolic_State<StateType>& state);

        /** Simplify current graph by removing unnecessary dummy nodes introduced
         *  in graph editing process without changing the grammar rules.
         *  Dummy nodes are internal structure used by this AOG library, doesn't
         *  corresponds to any grammar component.
        */
        void TruncateGraph();

        /**
         * Output a file that can be used for basic AOG visualization
         * @param:
         *      filename: name of the output file
         *      dir: name of the output file directory
         *      truncate: whether remove unnecessary dummy nodes
        */
        void Visualize(std::string dir, std::string filename, bool truncate = true);

        /**
         * Output a file that can be used to reconstruct an AOG
         * @param:
         *      path: output file path, doesn't include file name, as
         *            file name will be learned_tree_appendix.txt
         *      appendix: the appendix attached in the file name
         * @return:
         *      void
        */
        void SaveGraph(std::string path, std::string filename);
    };

    template<class StateType, class AttributeType>
    AOG<StateType, AttributeType>::AOG()
    :AOG_Graph<StateType, AttributeType>()
    { this->has_root_ = false; }
    
    //copy constructor for AOG
    template<class StateType, class AttributeType>
    AOG<StateType, AttributeType>::AOG(const AOG &other)
    :AOG_Graph<StateType, AttributeType>(other), all_rules_(other.all_rules_),
    all_leaf_states_(other.all_leaf_states_), state_to_vertex_(other.state_to_vertex_),
    root_id_(other.root_id_), has_root_(other.has_root_)
    {}

    template<class StateType, class AttributeType>
    AOG<StateType, AttributeType>::AOG(const std::vector<Symbolic_Rule<StateType> > &rules)
    :AOG_Graph<StateType, AttributeType>(0, true)
    {
        has_root_ = false;
        for (auto rule: rules)
            this->AddRule(rule);
    }

    template<class StateType, class AttributeType>
    AOG<StateType, AttributeType>::AOG(const std::vector<Symbolic_State<StateType> > &leaf_states)
    :AOG_Graph<StateType, AttributeType>(0, true)
    {
        //construct a root node
        has_root_ = false;
        std::shared_ptr<Symbolic_State<StateType> > root_state =
            std::make_shared<Symbolic_State<StateType> >();
        std::shared_ptr<AOG_Vertex<StateType, AttributeType> > root =
            std::make_shared<AOG_Vertex<StateType, AttributeType> >(*root_state, false, true);
        this->root_id_ = this->AddVertex(root);
        this->state_to_vertex_[*root_state] = this->root_id_;
        has_root_ = true;

        for (unsigned i = 0; i < leaf_states.size(); i++)
        {
            if (!leaf_states[i].GetIsBasic())
            {
                std::cerr << "Non basic states passed in\n";
                throw std::exception();
            }
            //duplicate states in the sequence
            if (this->all_leaf_states_.find(leaf_states[i]) != this->all_leaf_states_.end())
                continue;
            //take in leaf states as content, not an and-node, not a root node
            std::shared_ptr<AOG_Vertex<StateType, AttributeType> > ptr =
                std::make_shared<AOG_Vertex<StateType, AttributeType> >(leaf_states[i], false, false);
            VertexId new_v_id = this->AddVertex(ptr);
            this->all_leaf_states_.insert(leaf_states[i]);
            this->state_to_vertex_[leaf_states[i]] = new_v_id;
        }
    }

    template<class StateType, class AttributeType>
    AOG<StateType, AttributeType>::AOG(const std::string& file_path)
    {
        std::ifstream f(file_path, std::ifstream::in);
        if (f.is_open())
            std::cerr << "reading " << file_path << '\n';
        else
        {
            std::cerr << "file " << file_path <<  " cannot be found" << '\n';
            exit(1);
        }
        
        std::vector<Symbolic_Rule<StateType> > rules;

        std::string s = "";
        int line_num = 0;
        std::unordered_map<Symbolic_Rule<StateType>, double> m;
        while (getline(f, s))
        {
            line_num++;
            std::stringstream ss(s);
            std::string item;
            std::vector<std::string> tokens;
            while (getline(ss, item, ',')) 
                tokens.push_back(item);

            if (tokens.size() < 4)
            {
                std::cerr << "Error: row does not have source id and content\n";
                exit(1);
            }

            int num_res = stoi(tokens[0]);
            double weight = stod(tokens[1]);
            int src_id = stoi(tokens[2]);
            std::string src_ct = tokens[3];
            Symbolic_State<StateType> src(src_id);

            std::vector<Symbolic_State<StateType> > results;
            for (int i = 4; i < tokens.size(); i += 2)
            {
                Symbolic_State<StateType> state;
                int state_id = stoi(tokens[i]);
                if (state_id == -1)
                {
                    std::string state_ct = tokens[i+1];            
                    Symbolic_State<StateType> temp(state_ct, true);
                    state = temp;
                }
                else
                {
                    Symbolic_State<StateType> temp(state_id);
                    state = temp;
                }
                results.push_back(state);
            }
            Symbolic_Rule<StateType> rule(src, results);
            rules.push_back(rule);
            if(weight != 0)
                m[rule] = weight;
        }

        // Find root
        std::unordered_set<Symbolic_State<StateType> > top_level_rules;
        std::vector<Symbolic_State<StateType> > sources;
        std::unordered_set<Symbolic_State<StateType> > results;
        for (Symbolic_Rule<StateType> rule : rules)
        {
            sources.push_back(rule.GetSource());
            results.insert(rule.GetResults().begin(), rule.GetResults().end());
        }
        
        // check which source is not other sources' result
        for (Symbolic_State<StateType> source : sources)
        {
            if (results.find(source) == results.end())
                top_level_rules.insert(source);
        }
        
        if (top_level_rules.size() != 1)
        {
            std::cerr<<"top_level_rule: \n";
            for(const auto & state : top_level_rules)
                std::cerr<<"("<<state.GetContent()<<", "<<state.GetId()<<")";
            std::cerr << '\n' << "Dangling top level rules beside root\n";
            exit(1);
        }

        std::shared_ptr<AOG_Vertex<StateType, AttributeType> >
            source_ptr(new AOG_Vertex<StateType, AttributeType>(*top_level_rules.begin(), true, true));
        this->AddVertex(source_ptr);

        for (const auto & rule : rules)
            this->AddRule(rule);

        for(const auto& it : m)
        {
            const Symbolic_Rule<StateType>& rule = it.first;
            double weight = it.second;
            VertexId srcId = this->GetVertexIdByState(rule.GetSource());
            std::vector<VertexId> resIds;
            for (const Symbolic_State<StateType>& state : rule.GetResults())
                resIds.push_back(this->GetVertexIdByState(state));
            
            bool found = false;
            VertexId target_dummy;
            for(VertexId dummy : this->ChildrenVertices(srcId))
            {
                std::vector<VertexId> children = this->ChildrenVertices(dummy);
                if(children == resIds)
                {
                    found = true;
                    target_dummy = dummy;
                    break;
                }
            }
            if(!found)
            {
                std::cerr << "rule not found in graph !!" << '\n';
                exit(1);
            }
            auto weights = this->GetOutEdgeWeights(srcId, false);
            weights[target_dummy] = weight;
            this->SetOutEdgeWeights(srcId, weights);
        }
    }

    template<class StateType, class AttributeType>
    std::vector<Symbolic_Rule<StateType> > AOG<StateType, AttributeType>::GetRules() const
    {
        std::vector<Symbolic_Rule<StateType> > all_rules;
        for (auto rule: this->all_rules_)
            all_rules.push_back(rule);
        return all_rules;
    }

    template<class StateType, class AttributeType>
    std::vector<Symbolic_State<StateType> > AOG<StateType, AttributeType>::GetLeafStates() const
    {
        std::vector<Symbolic_State<StateType> > all_states(this->all_leaf_states_.size());
        for (auto state: this->all_leaf_states_)
            all_states.push_back(state);
        return all_states;
    }

    template<class StateType, class AttributeType>
    std::vector<Symbolic_State<StateType> > AOG<StateType, AttributeType>::GetStates() const
    {
        std::vector<Symbolic_State<StateType> > all_states(0);
        for (auto state: this->state_to_vertex_)
            all_states.push_back(state.first);
        return all_states;
    }

    template<class StateType, class AttributeType>
    VertexId AOG<StateType, AttributeType>::GetRoot() const 
    {
        if (!this->has_root_){
            std::cout << "The TAOG does not have a root\n";
            throw std::exception();
        }
        return this->root_id_;
    }

    template<class StateType, class AttributeType>
    Symbolic_State<StateType> AOG<StateType, AttributeType>::
    GetStateByVertexId(const VertexId vid) const
    {
        if (this->IsValidVertex(vid))
            return this->GetVertexContent(vid)->GetState();
        std::cerr<<"Invalid State\n";
        throw std::exception();
    }

    template<class StateType, class AttributeType>
    VertexId AOG<StateType, AttributeType>::
    GetVertexIdByState(const Symbolic_State<StateType> &state) const
    {
        auto iter = this->state_to_vertex_.find(state);
        if (iter == this->state_to_vertex_.end())
        {
            std::cerr << "Invalid State\n";
            throw std::exception();
        }
        return iter->second;
    }

    template<class StateType, class AttributeType>
    std::vector<Symbolic_Rule<StateType> > AOG<StateType, AttributeType>::
    GetRulesAsSource(const Symbolic_State<StateType> & query_state) const
    {
        std::vector<Symbolic_Rule<StateType> > rules;
        for(auto rule: this->all_rules_)
            if(rule.GetSource() == query_state)
                rules.push_back(rule);
        return rules;
    }

    template<class StateType, class AttributeType>
    std::vector<Symbolic_Rule<StateType> > AOG<StateType, AttributeType>::
    GetRulesAsTarget(const Symbolic_State<StateType> & query_state) const
    {
        std::vector<Symbolic_Rule<StateType> > rules;
        for(auto rule: this->all_rules_)
        {
            std::vector<Symbolic_State<StateType> > results = rule.GetResults();
            if(find(results.begin(), results.end(), query_state) != results.end())
                rules.push_back(rule);
        }
        return rules;
    }

    template<class StateType, class AttributeType>
    std::unordered_map<VertexId, double> AOG<StateType, AttributeType>::
    GetOutEdgeWeights(VertexId source, bool is_normalized) const
    {

        //report error if trying to get weights of an and-node's edges
        if(this->GetVertexContent(source)->IsAnd())
        {
            std::cerr<<"Cannot get weights of edges from an And-node\n";
            throw std::exception();
        }

        std::unordered_map<VertexId, double> weights;
        double total_weights = 0.0;
        for(auto edge:this->OutEdges(source))
        {
            double weight = this->GetEdgeContent(source, edge.second)[0]->GetWeight();
            total_weights += weight;
            weights[edge.second] = weight;
        }

        //if user wants unnormalized weights
        if(!is_normalized)
            return weights;

        double coeff = 1.0/total_weights;
        for(auto &iter : weights)
            iter.second *= coeff;

        return weights;
    }

    //T-AOG version add vertex, avoid adding duplicate vertices if they have same state
    template<class StateType, class AttributeType>
    VertexId AOG<StateType, AttributeType>::
    AddVertex(const std::shared_ptr<AOG_Vertex<StateType, AttributeType> > aog_vertex)
    {
        Symbolic_State<StateType> state = aog_vertex->GetState();
        bool is_and = aog_vertex->IsAnd();
        bool is_root = aog_vertex->IsRoot();
        //check if user wants to add a second root
        if(is_root && has_root_)
        {
            std::cerr<<"Cannot add a root vertex.\n";
            throw std::exception();
        }
        //if the symbolic state in the given vertex does not exist, add a new vertex
        auto iter = this->state_to_vertex_.find(state);
        
        if (iter == this->state_to_vertex_.end())
        {
            std::shared_ptr<AOG_Vertex<StateType, AttributeType> >
                ptr(new AOG_Vertex<StateType, AttributeType>(state, is_and, is_root));
            VertexId new_id = AOG_Graph<StateType, AttributeType>::AddVertex(ptr);
            this->state_to_vertex_[state] = new_id;
            if (state.GetIsBasic())
                this->all_leaf_states_.insert(ptr->GetState());
            
            //if the vertex added want to be a root, add this vertex as root
            if(is_root && !(this->has_root_))
            {
                this->has_root_ = true;
                this->root_id_ = new_id;
            }
            return new_id;
        }

        //else return the existing vertex's id
        if(is_root && !(this->has_root_))
        {
            this->has_root_ = true;
            this->root_id_ = iter->second;
        }
        return iter->second;
    }

    template<class StateType, class AttributeType>
    bool AOG<StateType, AttributeType>::DeleteVertex(const VertexId vid)
    {
        //update the data structures that keep track of all states and leaf states
        if (this->IsValidVertex(vid))
        {
            std::shared_ptr<AOG_Vertex<StateType, AttributeType> > aog_vertex = this->GetVertexContent(vid);
            Symbolic_State<StateType> state = aog_vertex->GetState();
            if (state.GetIsBasic())
                this->all_leaf_states_.erase(state);
            this->state_to_vertex_.erase(state);
        }

        //Invalid situation handled in base class
        return AOG_Graph<StateType, AttributeType>::DeleteVertex(vid);
    }


    template<class StateType, class AttributeType>
    bool AOG<StateType, AttributeType>::AddEdge(const VertexId source, const VertexId target,
                                                const std::shared_ptr<AOG_Edge> aog_edge,
                                                bool multi_edge, int pos)
    {
        double weight = aog_edge->GetWeight();
        //bound checking
        if(weight < 0)
        {
            std::cerr << "Weight cannot be less than 0!\n";
            return false;
        }
        if(this->IsValidVertex(source))
        {
            //source of the edge to add should not be a leaf state
            Symbolic_State<StateType> src_state = GetStateByVertexId(source);
            if(src_state.GetIsBasic())
            {
                std::cerr<<"cannot add edge starting from leaf state\n";
                return false;
            }
        }

        //invalid situations handled in base class
        std::shared_ptr<AOG_Edge> ptr = std::make_shared<AOG_Edge>(weight);
        return AOG_Graph<StateType, AttributeType>::AddEdge(source, target, ptr, multi_edge, pos);
    }

    template <class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::SetRoot(Symbolic_State<StateType> root)
    {
        this->has_root_ = true;
        this->root_id_ = this->GetVertexIdByState(root);
    }

    template<class StateType, class AttributeType>
    bool AOG<StateType, AttributeType>::
    SetOutEdgeWeights(VertexId source, const std::unordered_map<VertexId, double> & weights)
    {

        if(this->GetVertexContent(source)->IsAnd())
        {
            std::cerr<<"Cannot get weights of edges from an And-node\n";
            return false;
        }
        //check if the number of weights inputs is the same as the number of out edges
        if(weights.size() != this->OutEdges(source).size())
        {
            std::cerr<<"The number of weights inputs is inconsistent with the number of out edges.\n";
            return false;
        }
        //update weights
        for(auto iter:weights)
            this->GetEdgeContent(source, iter.first)[0]->SetWeight(iter.second);

        return true;
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::SetIsAnd(VertexId source_id, const bool is_and)
    {
        if (this->IsValidVertex(source_id))
            this->GetVertexContent(source_id)->SetIsAnd(is_and);
        else
            std::cerr << "Vertex" << source_id << " doesn't exist.\n";

        return;
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::SetVertexAttribute(VertexId source_id, const AttributeType& new_attribute)
    {
        if (this->IsValidVertex(source_id))
            this->GetVertexContent(source_id)->attributes_ = new_attribute;
        else
            std::cerr << "Vertex" << source_id << " doesn't exist.\n";

        return;
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::SetVertexPotential(VertexId source_id, const double new_potential)
    {
        if (this->IsValidVertex(source_id))
            this->GetVertexContent(source_id)->potential_ = new_potential;
        else
            std::cerr << "Vertex" << source_id << " doesn't exist.\n";

        return;
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::
    SetVertexPotentialFunc(VertexId source_id,
                           const std::function<double(AOG_Vertex<StateType, AttributeType>&,
                                                      const std::vector<std::shared_ptr<AOG_Vertex<StateType, AttributeType> > >&)> func)
    {
        if (this->IsValidVertex(source_id))
            this->GetVertexContent(source_id)->SetPotentialFunc(func);
        else
            std::cerr << "Vertex" << source_id << " doesn't exist.\n";

        return;
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::UpdateVertexPotential(VertexId v_id)
    {
        if (this->IsValidVertex(v_id))
            this->GetVertexContent(v_id)->UpdatePotential();
        else
            std::cerr << "Vertex " << v_id << " doesn't exist.\n";
        return;
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::AddVertexNeighbor(VertexId self_id, VertexId neighbor_id, bool check_dup)
    {
        if (this->IsValidVertex(self_id) && this->IsValidVertex(neighbor_id))
        {
            if(!check_dup)
                this->GetVertexContent(self_id)->neighbors_.push_back(this->GetVertexContent(neighbor_id));
            else
            {
                if(find(this->GetVertexContent(self_id)->neighbors_.begin(),
                        this->GetVertexContent(self_id)->neighbors_.end(),
                        this->GetVertexContent(neighbor_id)) != this->GetVertexContent(self_id)->neighbors_.end())
                    return;
                else
                    this->GetVertexContent(self_id)->neighbors_.push_back(this->GetVertexContent(neighbor_id));
            }
        }
        else
            std::cerr << "Vertex " << self_id << " or Vertex " << neighbor_id << " doesn't exist.\n";
        return;
    }

    template<class StateType, class AttributeType>
    bool AOG<StateType, AttributeType>::DeleteVertexNeighbor(VertexId self_id, VertexId neighbor_id)
    {
        if (this->IsValidVertex(self_id) && this->IsValidVertex(neighbor_id))
        {
            auto pos = find(this->GetVertexContent(self_id)->neighbors_.begin(),
                            this->GetVertexContent(self_id)->neighbors_.end(),
                            this->GetVertexContent(neighbor_id));
            if(pos == this->GetVertexContent(self_id)->neighbors_.end())
                return false;
            else
            {
                this->GetVertexContent(self_id)->neighbors_.erase(pos);
                return true;
            }
        }
        else
        {
            std::cerr << "Vertex " << self_id << " or Vertex " << neighbor_id << " doesn't exist.\n";
            exit(1);
        };
    }

    template<class StateType, class AttributeType>
    std::unordered_map<VertexId,double> AOG<StateType, AttributeType>::Normalize(VertexId src_id)
    {
        //if it is an And-node, do nothing
        if(this->GetVertexContent(src_id)->IsAnd())
        {
            std::cerr << "Cannot normalize edges from And-node\n";
            throw std::exception();
        }

        //get normalized weights
        std::unordered_map<VertexId,double> weights = this->GetOutEdgeWeights(src_id,true);
        //set weights
        this->SetOutEdgeWeights(src_id,weights);
        return weights;
    }
    
    template<class StateType, class AttributeType>
    bool AOG<StateType, AttributeType>::ExistRule(const Symbolic_Rule<StateType>& rule)
    {
        return this->all_rules_.find(rule) != this->all_rules_.end();
    }

    template<class StateType, class AttributeType>
    bool AOG<StateType, AttributeType>::AddRule(const Symbolic_Rule<StateType>& rule)
    {
        // construct an edge with weight 1.0
        std::shared_ptr<AOG_Edge> edge = std::make_shared<AOG_Edge>();

        // check if the rule already exists
        if (this->ExistRule(rule))        
            return false;

        // get the source and result states from each rule and construct vertices for each of them.
        std::shared_ptr<Symbolic_State<StateType> > source_state = 
            std::make_shared<Symbolic_State<StateType> >(rule.GetSource());
        std::shared_ptr<AOG_Vertex<StateType, AttributeType> > source = 
            std::make_shared<AOG_Vertex<StateType, AttributeType> >(*source_state, true, false);
        
        // std::cerr<<"this rule is added from state id: "<<(*source_state).GetId()<<std::endl;
        VertexId src_id = this->AddVertex(source);               
        this->state_to_vertex_[*source_state] = src_id;
    
        std::vector<std::shared_ptr<Symbolic_State<StateType> > > result_states;
        std::vector<VertexId> rs_vtxs_id;
        auto results = rule.GetResults();

        for(auto result : results)
        {
            // std::cerr<<"this state is added to state id: "<<result.GetId();
            result_states.push_back(std::make_shared<Symbolic_State<StateType> >(result));
            std::shared_ptr<AOG_Vertex<StateType, AttributeType> > rs_vtx = 
                std::make_shared<AOG_Vertex<StateType, AttributeType> >(result,true,false);
            VertexId rs_id = this->AddVertex(rs_vtx);
            rs_vtxs_id.push_back(rs_id);
            if(result.GetIsBasic())
                this->all_leaf_states_.insert(result);
            this->state_to_vertex_[result] = rs_id;
        }
                       
        std::vector<VertexId> children = AOG_Graph<StateType, AttributeType>::ChildrenVertices(src_id);


        //create a dummy And-node
        std::shared_ptr<Symbolic_State<StateType> > dummy_state = 
            std::make_shared<Symbolic_State<StateType> >(*source_state);
        std::shared_ptr<AOG_Vertex<StateType, AttributeType> > dummy = 
            std::make_shared<AOG_Vertex<StateType, AttributeType> >(*dummy_state, true, false);
        // if src is already a parent
        if (children.size()) {
            auto src_content = source_state->GetContent();

            // Add the dummy And-node for the new rule
            VertexId dummy_id = AOG_Graph<StateType, AttributeType>::AddVertex(dummy);

            // if src is an And-node, it has no dummy nodes yet
            if (this->GetVertexContent(src_id)->IsAnd())
            {
                // create an Or-node new parent between src and its parent
                std::shared_ptr<Symbolic_State<StateType> > new_par_state =
                    std::make_shared<Symbolic_State<StateType> >(*source_state);
                std::shared_ptr<AOG_Vertex<StateType, AttributeType> > new_par =
                    std::make_shared<AOG_Vertex<StateType, AttributeType> >(*new_par_state, false, false);
                //delete the old state and add the new state
                state_to_vertex_.erase(*source_state);
                VertexId new_par_id = this->AddVertex(new_par);
                this->state_to_vertex_[*new_par_state] = new_par_id;

                //change the root to the newly created node
                if(this->has_root_ && src_id == this->GetRoot())
                {
                    this->root_id_ = new_par_id;
                    this->GetVertexContent(src_id)->SetIsRoot(false);
                    this->GetVertexContent(new_par_id)->SetIsRoot(true);

                }

                // if src has parents, connect parents to new_par
                // delete parents to src
                for (auto parent_id : this->ParentsVertices(src_id)) 
                {                        
                    //find the source node's positions in all rules whose targets contains source
                    std::vector<VertexId> all_children = this->ChildrenVertices(parent_id);                
                    std::vector<int> result_pos;
                    auto result = std::find(all_children.begin(),all_children.end(),src_id);
                    while(result != all_children.end())
                    {
                        result_pos.push_back(result - all_children.begin());
                        result = std::find(result + 1,all_children.end(),src_id);
                    }
                    
                    //delete all edges that have given source and target
                    this->DeleteEdge(parent_id, src_id);

                    //add all edges from parent to newly added state
                    for(int pos : result_pos)
                        this->AddEdge(parent_id, new_par_id, edge, true, pos);

                }
                // connect new_par to src and dummy
                this->AddEdge(new_par_id, src_id, edge);
                this->AddEdge(new_par_id, dummy_id, edge);
                // connect dummy to all results
                for(auto rs_id : rs_vtxs_id)
                    this->AddEdge(dummy_id, rs_id, edge);

            }
            // if src is an Or-node, simply add another dummy node
            else
            {
                // connect src to dummy
                this->AddEdge(src_id, dummy_id, edge);
                // connect dummy to all results
                for(auto rs_id : rs_vtxs_id)
                    this->AddEdge(dummy_id, rs_id, edge);
            }
        }

        // if src is not yet a parent, and src is root,
        // add a dummy node below, set root to or_node, then add children under dummy  
        else if(!children.size() && this->GetVertexContent(src_id)->IsRoot()) 
        {
            VertexId dummy_id = AOG_Graph<StateType, AttributeType>::AddVertex(dummy);
            this->AddEdge(src_id,dummy_id,edge);
           
            for(auto rs_id : rs_vtxs_id)
                this->AddEdge(dummy_id, rs_id, edge);
            
            //change root to or-node
            this->GetVertexContent(src_id)->SetIsAnd(false);

        }

        // if src is not yet a parent, and src is not root, just add edges to its children
        else
        {
            // std::cerr<<"the edge is added from: "<<src_id<<std::endl;
            for(auto rs_id : rs_vtxs_id)
            {
                // std::cerr<<"the edge is added to : "<<rs_id<<std::endl;
                this->AddEdge(src_id, rs_id, edge);
            }
            
        }

        this->all_rules_.insert(rule);
        return true;
    }
    
    template<class StateType, class AttributeType>
    bool AOG<StateType, AttributeType>::DeleteRule(const Symbolic_Rule<StateType>& rule)
    {
      //delete the rule in all_rules_
        if(!all_rules_.erase(rule))
        {
            //std::cerr<<"The rule to be deleted does not exist in the AOG!\n";
            return false;
        }
    
        //get vertices ids
        VertexId src_vtx = GetVertexIdByState(rule.GetSource());
        auto results = rule.GetResults();
        std:: vector<VertexId> end_vtxs;
        for (auto result : results)
            end_vtxs.push_back(GetVertexIdByState(result));
    
        size_t end_vtxs_size = end_vtxs.size();
    

        //if the source node is an and-node
        if(this->GetVertexContent(src_vtx)->IsAnd())
        {
            //delete the edges constructed from the rule
            for(auto end_vtx : end_vtxs)
                this->DeleteEdge(src_vtx,end_vtx);
        }

        //if source state is an or-node
        else
        {
            //find the dummy and delete it
            auto dummys = this->ChildrenVertices(src_vtx);
            bool found_rule;
            for(auto dummy : dummys)
            {
                found_rule = true;
                auto dummy_children = this->ChildrenVertices(dummy);
        
                if(dummy_children.size() != end_vtxs_size)
                    continue;
                //if the child is the dummy node that associated with the rule, delete the dummy
                for(size_t i = 0; i< end_vtxs_size;i++)
                    if(dummy_children[i] != end_vtxs[i])
                    {
                        found_rule = false;
                        break;
                    }
                if(found_rule)
                {
                    AOG_Graph<StateType, AttributeType>::DeleteVertex(dummy);
                    break;
                }
            }

            //if there is only one dummy left, and the source is not root, change the source vertex to an and-node
            dummys = this->ChildrenVertices(src_vtx);
            if(dummys.size() == 1 && (!this->GetVertexContent(src_vtx)->IsRoot()))
            {
                std::vector<VertexId> dummy_children = this -> ChildrenVertices(dummys[0]);
                for(auto dummy_child : dummy_children)
                    AddEdge(src_vtx,dummy_child,std::make_shared<AOG_Edge>());
                //delete the dummy node
                AOG_Graph<StateType, AttributeType>::DeleteVertex(dummys[0]);
        
                //change the property of the source vertex to and-node
                this->GetVertexContent(src_vtx)->SetIsAnd(true);
            }
        }
        return true;
    }
  
    template<class StateType, class AttributeType>
    std::shared_ptr<std::vector<VertexId>> AOG<StateType, AttributeType>::
    Sample(VertexId root, std::vector<VertexId>& res, double &prob) const
    {
        prob = 1;
        //if the node to sample from is a dummy node, start sampling from its parent
        auto root_parent = this->ParentsVertices(root);
        if(!root_parent.empty() && !this->GetVertexContent(root_parent[0])->IsAnd())
            root = root_parent[0];
        // the resultant graph that contains the key-value pairs
        // std::unordered_map<VertexId, std::vector<VertexId> > sample_graph;
        std::shared_ptr<std::vector<VertexId>> parse_tree = std::make_shared<std::vector<VertexId> >();
        // for level-order tree traversal
        std::stack<VertexId> stack;
        // will be used to obtain a seed for the random number engine
        std::random_device rd;
        // standard mersenne_twister_engine seeded with rd()
        std::mt19937 gen(rd());

        stack.push(root);
        while (!stack.empty())
        {
            VertexId parent_id = stack.top();
            stack.pop();
            parse_tree->push_back(parent_id);
            std::vector<VertexId> children = this->ChildrenVertices(parent_id);

            if (children.size() == 0)
                res.push_back(parent_id);
            
            // if the node is an And-node, keep all its children in sample_graph
            else if (AOG_Graph<StateType, AttributeType>::GetVertexContent(parent_id)->IsAnd())
            {
                for (int i = children.size()-1; i >= 0; i--)
                    stack.push(children[i]);
            }
            // if the node is an Or-node
            else
            {
                std::unordered_map<VertexId, double> edge_weights = this->GetOutEdgeWeights(parent_id, true);
                
                std::vector<VertexId> vertex;
                std::vector<double> weight;

                for (auto w:edge_weights)
                {
                    vertex.push_back(w.first);
                    weight.push_back(w.second);
                }

                // choose a child under the weight distribution and keep it in sample_graph
                std::discrete_distribution<> dis(weight.begin(), weight.end());
                unsigned index = dis(gen);
                VertexId sample = vertex[index];
                prob *= weight[index];
                
                stack.push(sample);
            }
        }

        return parse_tree;
    }

    template<class StateType, class AttributeType>    
    std::vector<Symbolic_State<StateType> > AOG<StateType, AttributeType>::GetTopLevelStates()
    {
        std::vector<Symbolic_State<StateType> > top_level_states;
        std::unordered_set<Symbolic_State<StateType> > sources;
        std::unordered_set<Symbolic_State<StateType> > results;
        for (Symbolic_Rule<StateType> rule : this->GetRules())
        {
            sources.insert(rule.GetSource());
            results.insert(rule.GetResults().begin(), rule.GetResults().end());
        }

        // check which source is not other sources' result
        for (Symbolic_State<StateType> source : sources)
        {
            if (results.find(source) == results.end())
                top_level_states.push_back(source);
        }
        return top_level_states;
    }

    template <class StateType, class AttributeType>
    double AOG<StateType, AttributeType>:: GrammarSize(double leaf_bias)
    {
        double grammar_size = 0;
        for(const auto& rule : this->GetRules())
        {
            ++grammar_size;
            for(const auto& state : rule.GetResults())
            {
                if(state.GetId() == -1)
                    grammar_size += leaf_bias;
                else    
                    ++grammar_size;
            }
        }
        return grammar_size;
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::TruncateGraph()
    {
        VertexId root_id = this->GetRoot();
        std::queue<VertexId> traverse_q;
        traverse_q.push(root_id);
        int count = 1;
        while (!traverse_q.empty())
        {
            VertexId cur_vertex_id = traverse_q.front();
            traverse_q.pop();

            std::vector<VertexId> children_id = this->ChildrenVertices(cur_vertex_id);
            for (VertexId child_id : children_id)
            {
                traverse_q.push(child_id);
            }

            if (children_id.size() == 1 && cur_vertex_id != this->GetRoot())
            {
                // find its parent, and connect child to parent
                std::vector<VertexId> parents_id = this->ParentsVertices(cur_vertex_id);

                // the reconnection is needed for all parents of cur_vertex_id
                for (VertexId p_id : parents_id)
                {
                    // delete old edge parent to cur
                    std::vector<VertexId> p_children = this->ChildrenVertices(p_id);
                    int pos = -1;
                    std::vector<std::vector<std::shared_ptr<AOG_Edge> > > parent_to_p;
                    for (int i = 0; i < p_children.size(); i++)
                    {
                        if (p_children[i] == cur_vertex_id)
                            pos = i;
                        parent_to_p.push_back(this->GetEdgeContent(p_id, p_children[i]));
                        this->DeleteEdge(p_id, p_children[i]);
                    }

                    for (int i = 0; i < pos; i++)
                    {
                        std::shared_ptr<AOG_Edge> new_edge = std::make_shared<AOG_Edge>();                        
                        // Or node will not have multi-edge
                        if (!this->GetVertexContent(p_id)->IsAnd())
                        {
                            if (parent_to_p[i].size() == 0)
                                throw std::exception();
                            if (parent_to_p[i].size() > 1)
                            {
                                std::cerr << "Multiedge\n";
                                throw std::exception();
                            }
                            new_edge->SetWeight(parent_to_p[i][0]->GetWeight());
                        }
                        // add new edge with weight if parent is Or
                        this->AddEdge(p_id, p_children[i], new_edge);
                    }

                    // repeat same operation for cur_vertex_id
                    // if parent is Or-node, copy its weight when creating the edge
                    // Or node will not have multi-edge
                    std::shared_ptr<AOG_Edge> new_edge = std::make_shared<AOG_Edge>();                        
                    if (!this->GetVertexContent(p_id)->IsAnd())
                    {
                        if (parent_to_p[pos].size() == 0)
                            throw std::exception();
                        if (parent_to_p[pos].size() > 1)
                        {
                            std::cerr << "Multiedge\n";
                            throw std::exception();
                        }
                        new_edge->SetWeight(parent_to_p[pos][0]->GetWeight());
                    }
                    // add new edge with weight if parent is Or
                    this->AddEdge(p_id, children_id[0], new_edge);

                    for (int i = pos+1; i < p_children.size(); i++)
                    {
                        std::shared_ptr<AOG_Edge> new_edge = std::make_shared<AOG_Edge>();                                                
                        // Or node will not have multi-edge
                        if (!this->GetVertexContent(p_id)->IsAnd())
                        {
                            if (parent_to_p[i].size() == 0)
                                throw std::exception();
                            if (parent_to_p[i].size() > 1)
                            {
                                std::cerr << "Multiedge\n";
                                throw std::exception();
                            }
                            new_edge->SetWeight(parent_to_p[i][0]->GetWeight());
                        }
                        // add new edge with weight if parent is Or
                        this->AddEdge(p_id, p_children[i], new_edge);
                    }
                }
                // delete old edge cur to child
                this->DeleteEdge(cur_vertex_id, children_id[0]);
                // delete dummy node
                this->DeleteVertex(cur_vertex_id);
            }
        }
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::DeleteNoParentRules(const Symbolic_State<StateType>& src_state)
    {
        VertexId vid = this -> GetVertexIdByState(src_state);
        
        //if the passed in vertex has parent, is root, or the state is a leaf, directly return
        if(this->ParentsVertices(vid).size() || src_state.GetIsBasic() || vid == this->root_id_)
            return;
        std::vector<VertexId> children_vtx = this->ChildrenVertices(vid);
        // if it is an and-node
        if(this->GetVertexContent(vid)->IsAnd())
        {
            //delete curent rule
            std::vector<Symbolic_State<StateType> > children_states;
            for(auto vertex : children_vtx )
                children_states.push_back(this->GetStateByVertexId(vertex));
            
            if(!this->DeleteRule(Symbolic_Rule<StateType>(src_state ,children_states)))
                std::cerr<<"fail to delete rules in and-node:\n";
            
            //recursively delete child states if they have no parents
            for(auto child_state : children_states)
                this->DeleteNoParentRules(child_state);
        }
        //if it is an or-node
        else
        {
            std::vector< std::vector< Symbolic_State<StateType> > > grand_children_states;
            //first create all rules
            for(auto child : children_vtx)            
            {
                std::vector<VertexId> grand_children_vtx = this->ChildrenVertices(child);
                std::vector<Symbolic_State<StateType> > seq;
                for(auto grand_child : grand_children_vtx )
                    seq.push_back(this->GetStateByVertexId(grand_child));
                grand_children_states.push_back(seq);
            }
            //delete all rules
            for(int i = 0; i < grand_children_states.size(); ++i)
            {
                if(!this->DeleteRule(Symbolic_Rule<StateType>(src_state ,grand_children_states[i])))
                    std::cerr<<"fail to delete rules in or-node\n";
                for(auto state : grand_children_states[i])
                    this->DeleteNoParentRules(state);
            }   
        }
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::Visualize(std::string dir, std::string filename, bool truncate)
    {
        std::cout << "PRINTED GRAPH\n";
        std::shared_ptr<AOG<StateType, AttributeType> > printed_graph =
            std::make_shared<AOG<StateType, AttributeType> >(*this);

        if (truncate)
            printed_graph->TruncateGraph();

        std::string dir_copy = dir;
        std::string filename_copy = filename;
        std::ofstream file;
        if (dir_copy.empty())
            dir_copy = filename_copy;
        else
            dir_copy = dir_copy.append("/").append(filename_copy);

        file.open(dir_copy, std::ofstream::out|std::ofstream::trunc);
        if (file.is_open()) {

            VertexId root_id = printed_graph->GetRoot();
            std::queue<VertexId> q;
            q.push(root_id);
            std::queue<unsigned> level_q;
            level_q.push(1);

            std::unordered_set<VertexId> plot_visited;

            while (!q.empty())
            {
                VertexId cur_vertex = q.front();
                unsigned level = level_q.front();
                q.pop();
                level_q.pop();

                std::vector<VertexId> children_vertices_id = printed_graph->ChildrenVertices(cur_vertex);

                if(!printed_graph->GetVertexContent(cur_vertex))
                {
                    int a;
                }
                bool isAnd = printed_graph->GetVertexContent(cur_vertex)->IsAnd();
                std::unordered_map<VertexId, double> weights;
                Symbolic_State<StateType> cur_vertex_state = printed_graph->GetStateByVertexId(cur_vertex);
                
                if (!isAnd)
                    weights = printed_graph->GetOutEdgeWeights(cur_vertex, true);
                    
                for (int i = 0; i < children_vertices_id.size(); i++)
                {
                    Symbolic_State<StateType> child_state = printed_graph->GetStateByVertexId(children_vertices_id[i]);
                    if (isAnd)
                    {    
                        file << cur_vertex << "," << children_vertices_id[i]
                             << "," << cur_vertex_state.GetContent() << "_"
                             << cur_vertex_state.GetId() << "," << child_state.GetContent()
                             << "_" << child_state.GetId() << ",," << i+1 << "," << level << "\n";
                    }
                    else
                    {
                        file << cur_vertex << "," << children_vertices_id[i]
                             << "," << cur_vertex_state.GetContent() << "_"
                             << cur_vertex_state.GetId() << "," << child_state.GetContent()
                             << "_" << child_state.GetId() << "," << weights[children_vertices_id[i]]
                             << "," << i+1 <<"," << level << "\n";
                    }
                    q.push(children_vertices_id[i]);
                    level_q.push(level+1);
                }
            }
            file.close();
        }
        else
            std::cout << "Unable to open " << dir_copy << std::endl;
    }

    template<class StateType, class AttributeType>
    void AOG<StateType, AttributeType>::SaveGraph(std::string path, std::string name)
    {
        std::vector<Symbolic_Rule<StateType>> rules = this->GetRules();
        std::string file_name;

        file_name = path + '/' + name;
        
        std::ofstream file;
        file.open(file_name, std::ofstream::out|std::ofstream::trunc);

        if (file.is_open())
        {
            std::cerr << "File opened. \n";
            for (const Symbolic_Rule<StateType>& rule : rules)
            {
                // size of result, souce id, source content, [child id, child content]*
                file << rule.GetResults().size();
                std::vector<Symbolic_State<StateType> > states = rule.GetResults();
                VertexId source_vtx_id = this->GetVertexIdByState(rule.GetSource());
                bool isAnd = this->GetVertexContent(source_vtx_id)->IsAnd();
                if(!isAnd){
                    std::vector<VertexId> children_vector;
                    for(const Symbolic_State<StateType>& state : states)
                        children_vector.push_back(this->GetVertexIdByState(state));
                    VertexId tmpDummy;
                    bool found = false;
                    for (VertexId dummy : this ->ChildrenVertices(source_vtx_id))
                    {
                        if (this->ChildrenVertices(dummy) == children_vector)
                        {
                            found = true;
                            tmpDummy = dummy;
                            break;
                        }
                    }
                    if (!found)
                    {
                        std::cerr << "rule not found" << std::endl;
                        throw std::exception();
                    }
                    double weight = this->GetOutEdgeWeights(source_vtx_id, false)[tmpDummy];
                    if (weight > 0)
                        file << "," << weight;
                    else{
                        std::cerr << "0 or negative weight is found !!" << std::endl;
                        throw std::exception();
                    }
                }
                else{
                    file << "," << 0;
                }
                file << "," << rule.GetSource().GetId() << "," << rule.GetSource().GetContent();
                for (int i = 0; i < states.size(); i++)
                {
                    file << "," << states[i].GetId() << "," << states[i].GetContent();
                }
                file << "\n";
            }

            file.close();
        }
        else
            std::cout << "Unable to open " << "learned_tree.txt" << std::endl;
    }
}

#endif//AOG_LIB_AOG_H
