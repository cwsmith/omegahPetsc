#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <sstream>
#include <iostream>
#include <string>
#include <algorithm>
using namespace Omega_h;

int main(int argc, char **argv)
{
  auto lib = Library(&argc, &argv);
  if(argc!=3) {
    fprintf(stderr, "Usage: %s <input mesh> <output prefix>\n", argv[0]);
    return 0;
  }

  const auto inmesh = argv[1];
  Mesh mesh(&lib);
  binary::read(inmesh, lib.world(), &mesh);

  const auto dim = mesh.dim();
  auto nvert = mesh.nverts(); 
  mesh.add_tag<LO>(0, "isOnMdlBdry", 0);
  Write<LO> isBdry(nvert, 0, "isBdry");

  Read<LO> edges_verts = mesh.get_adj(1, 0).ab2b;
  Read<LO> elem_edges = mesh.get_adj(dim, 1).ab2b;

  // Sort the element to edge adjacency to reveal which edge has 1 face
  std::vector<int> sorted_elem_edges(elem_edges.begin(), elem_edges.end());
  std::sort(sorted_elem_edges.begin(), sorted_elem_edges.end());
  std::vector<int> boundary_edge;
  for (unsigned int i = 0; i < sorted_elem_edges.size()-1; i++)
  {
    if (sorted_elem_edges[i] != sorted_elem_edges[i+1])
    {
      boundary_edge.push_back(sorted_elem_edges[i]);
    }
    else
    {
      i++;
    }
  }

  // Get the boundary vertices from the boundary edges
  for (unsigned int i = 0; i < boundary_edge.size(); i++)
  {
    int vert1 = edges_verts[2*boundary_edge[i]];
    int vert2 = edges_verts[2*boundary_edge[i]+1];
    isBdry[vert1] = 1;
    isBdry[vert2] = 1;
  }

  Read<LO> isBdry_r(isBdry);
  mesh.add_tag<LO>(0, "isOnMdlBdry", 1);
  mesh.set_tag<LO>(0, "isOnMdlBdry", isBdry_r);

  std::stringstream ss;
  ss << argv[2] << ".osh";
  std::string str = ss.str();
  binary::write(str, &mesh);
  ss.str("");
  ss << argv[2] << ".vtu";
  str = ss.str();
  vtk::write_vtu(str, &mesh);

  return 0;
}
