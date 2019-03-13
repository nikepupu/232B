//
// Created by yuanluyao on 10/23/17.
//

#ifndef AOG_LIB_AOG_VERTEX_H
#define AOG_LIB_AOG_VERTEX_H

#include <string>
#include <functional>
#include "Symbolic_State.h"
#include "../Core/Graph.hpp"
namespace AOG_LIB
{
    //forward declaration
    template <class StateType, class AttributeType> class AOG;
    class AOG_Edge;

    /* This class defines vertices in an AOG graph.
     * Mutators in this class are set as private to prevent unwanted changes of vertices' properties from users
     * This class declares T_AOG as friend class to enable T_AOG to change its vertices' properties.
     */
    template<class StateType, class AttributeType>
    class AOG_Vertex
    {
        bool is_and_; // true if the AOG Vertex is an And-node
        bool is_root_;// true if the AOG Vertex is a root node
        Symbolic_State<StateType> state_; // The Symbolic state contained in the AOG Vertex
        //The potential for this vertex calculated recursively
        double potential_;
        //The neighbors of this vertex that contribute to its potential
        std::vector<std::shared_ptr<AOG_Vertex> > neighbors_;
        //potential update function
        std::function<double(AOG_Vertex&, const std::vector<std::shared_ptr<AOG_Vertex> >&)> PotentialFunction_ = 0;
        //attributes of this vertex
        AttributeType attributes_;

        friend class AOG<StateType, AttributeType>;

        /* This function sets the symbolic state of a AOG Vertex.
         * @param:
         *      state: The symbolic state to be set
         * @return: void
         */
        void SetState(const Symbolic_State<StateType> &state){this->state_ = state;};
        
        /* This function sets a vertex to be an and-node or an or-node in AOG graph
         * @param:
         *      is_and: a boolean value that specifies whether the vertex is an and-node
         * @return: void
         */
        void SetIsAnd(const bool is_and){this->is_and_ = is_and;};

        /**
         * This function sets a vertex's potential calculate function
         * @param:
         *      func: the function to be used as the potential function
         * @return:
         *      void
        */
        void SetPotentialFunc(const std::function<double(AOG_Vertex&,
                                                         const std::vector<std::shared_ptr<AOG_Vertex> >&)> func)
        {this->PotentialFunction_ = func;};

        /**
         * This function sets a vertex to be a root or not root
         * @param:
         *      is_root: a boolean value that specifies whether the vertex is the root
         * @return:
         *      void
        */
        void SetIsRoot(bool is_root){this->is_root_ = is_root;};

        /**
         * This function updates potential of this vertex according to its neighbors
         * @param:
         *      void
         * @return:
         *      void
        */
        virtual void UpdatePotential();
    public:
        /* Construct an AOG Vertex from a symbolic state along with the node's properties.
         * @param:
         *      state: the symbolic state used to construct the vertex
         *      bool is_and: specify whether the vertex is an and-node or an or-node
         *      bool is_root: specify whether the vertex is a root node
         */
        AOG_Vertex(const Symbolic_State<StateType> &, bool, bool,
                   const std::function<double(AOG_Vertex&,
                                              const std::vector<std::shared_ptr<AOG_Vertex> >&)> p_func = 0);

        // return the current symbolic state of the AOG vertex
        const Symbolic_State<StateType>& GetState();

        // return a boolean value indicating whether the vertex is an and-node.
        bool IsAnd() const;

        // return a boolean value indicating whether the vertex is a root node.
        bool IsRoot() const;

        // return current potential of this vertex
        double GetPotential() const;
        
        // return current attribute of this vertex
        AttributeType GetAttribute() const;
    };


    template<class StateType, class AttributeType>
    AOG_Vertex<StateType, AttributeType>::AOG_Vertex(const Symbolic_State<StateType> &state, bool is_and, bool is_root,
                                                     const std::function<double(AOG_Vertex&,
                                                                                const std::vector<std::shared_ptr<AOG_Vertex> >&)> p_func)
    :state_(state), is_and_(is_and), is_root_(is_root), potential_(0), PotentialFunction_(p_func)
    {}

    template<class StateType, class AttributeType>
    void AOG_Vertex<StateType, AttributeType>::UpdatePotential()    
    {
        if(!PotentialFunction_)
        {
            std::cerr << "Vertex doesn't have potential function\n";
            exit(1);
        }
        this->potential_ = this->PotentialFunction_(*this, this->neighbors_);
    }

    template<class StateType, class AttributeType>
    const Symbolic_State<StateType>& AOG_Vertex<StateType, AttributeType>::GetState() 
    { return this->state_; }

    template<class StateType, class AttributeType>
    bool AOG_Vertex<StateType, AttributeType>::IsAnd() const
    { return this->is_and_; }

    template<class StateType, class AttributeType>
    bool AOG_Vertex<StateType, AttributeType>::IsRoot() const
    { return this->is_root_; }

    template<class StateType, class AttributeType>
    double AOG_Vertex<StateType, AttributeType>::GetPotential() const
    { return this->potential_; }

    template<class StateType, class AttributeType>
    AttributeType AOG_Vertex<StateType, AttributeType>::GetAttribute() const
    { return this->attributes_; }
    
}
#endif //AOG_LIB_AOG_VERTEX_H
