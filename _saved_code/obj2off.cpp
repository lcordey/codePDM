// Load the data from the .obj then convert it into .off

// Iostream - STD I/O Library
#include <iostream>

// fStream - STD File I/O Library
#include <fstream>

// OBJ_Loader - .obj Loader
#include "OBJ_Loader.h"

// Main function
int main(int argc, char *argv[])
{
	// Initialize Loader
	objl::Loader Loader;

	// Load .obj File
	bool loadout = Loader.LoadFile(argv[1]);

	// Check to see if it loaded

	// If so continue
	if (loadout)
	{
		int sum_vertices = 0;
		int sum_indices = 0;

		std::string vertices = "";
		std::string indices = "";
		
		// Go through each loaded mesh and out its contents
		for (int i = 0; i < Loader.LoadedMeshes.size(); i++)
		{
			// Copy one of the loaded meshes to be our current mesh
			objl::Mesh curMesh = Loader.LoadedMeshes[i];

			for (int j = 0; j < curMesh.Vertices.size(); j++)
			{
				vertices.append(std::to_string(curMesh.Vertices[j].Position.X));
				vertices.append(" ");
				vertices.append(std::to_string(curMesh.Vertices[j].Position.Y));
				vertices.append(" ");
				vertices.append(std::to_string(curMesh.Vertices[j].Position.Z));
				vertices.append("\n");
			}

			for (int j = 0; j < curMesh.Indices.size(); j += 3)
			{
				indices.append("3 ");
				indices.append(std::to_string(sum_vertices + curMesh.Indices[j]));
				indices.append(" ");
				indices.append(std::to_string(sum_vertices + curMesh.Indices[j+1]));
				indices.append(" ");
				indices.append(std::to_string(sum_vertices + curMesh.Indices[j+2]));
				indices.append(" ");
				indices.append(std::to_string((int)(curMesh.MeshMaterial.Kd.X * 255)));
				indices.append(" ");
				indices.append(std::to_string((int)(curMesh.MeshMaterial.Kd.Y * 255)));
				indices.append(" ");
				indices.append(std::to_string((int)(curMesh.MeshMaterial.Kd.Z * 255)));
				indices.append("\n");
			}
			
			sum_vertices += curMesh.Vertices.size();
			sum_indices += curMesh.Indices.size();
		}

		// Create/Open test.off and write results on it
		std::ofstream out(argv[2]);
		out << "OFF\n";
		out << sum_vertices << " " << sum_indices/3 << " " << "0\n";
		out << vertices;
		out << indices;

		// Close File
		out.close();
	}
}