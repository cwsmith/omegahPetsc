#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
using namespace Omega_h;

int main(int argc, char **argv)
{
  auto lib = Library(&argc, &argv);
  if(argc!=3) {
    fprintf(stderr, "Usage: %s <input mesh> <output prefix>\n", argv[0]);
    return 0;
  }
  const auto rank = lib.world()->rank();
  const auto inmesh = argv[1];
  Mesh mesh(&lib);
  binary::read(inmesh, lib.world(), &mesh);
  const auto dim = mesh.dim();

  auto nvert = mesh.nverts(); 
  mesh.add_tag<LO>(0, "isOnMdlBdry", 0);
  Write<LO> isBdry(nvert, 0, "isBdry");
  //loop here over adjacencies
  //
  //
  //
  Read<LO> isBdry_r(isBdry);
  mesh.set_tag<LO>(0, "isOnMdlBdry", isBdry_r);
  std::stringstream ss;
  ss << argv[2] << ".osh";
  std::string str = ss.str();
  binary::write(str, &mesh);
  ss.str(); //clear?
  ss << argv[2] << ".vtu";
  str = ss.str();
  vtk::write_vtu(str, &mesh);
  return 0;
}
