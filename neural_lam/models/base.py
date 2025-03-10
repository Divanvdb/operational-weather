# Third-party
import torch_geometric as pyg

# Local
from .. import utils
from ..config import NeuralLAMConfig
from ..datastore import BaseDatastore
from .base_model import BaseModel


class Base(BaseModel):
    """
    Full graph-based LAM model that can be used with different
    (non-hierarchical )graphs. Mainly based on GraphCast, but the model from
    Keisler (2022) is almost identical. Used for GC-LAM and L1-LAM in
    Oskarsson et al. (2023).
    """

    def __init__(self, args, config: NeuralLAMConfig, datastore: BaseDatastore):
        super().__init__(args, config=config, datastore=datastore)

        assert (
            not self.hierarchical
        ), "GraphLAM does not use a hierarchical mesh graph"

        # grid_dim from data + static + batch_static
        mesh_dim = self.mesh_static_features.shape[1]
        m2m_edges, m2m_dim = self.m2m_features.shape
        print(
            f"Edges in subgraphs: m2m={m2m_edges}, g2m={self.g2m_edges}, "
            f"m2g={self.m2g_edges}"
        )


    def get_num_mesh(self):
        """
        Compute number of mesh nodes from loaded features,
        and number of mesh nodes that should be ignored in encoding/decoding
        """
        return self.mesh_static_features.shape[0], 0

    def embedd_mesh_nodes(self):
        """
        Embed static mesh features
        Returns tensor of shape (N_mesh, d_h)
        """
        return self.mesh_embedder(self.mesh_static_features)  # (N_mesh, d_h)


    def process_step(self, mesh_rep):
        """
        Process step of embedd-process-decode framework
        Processes the representation on the mesh, possible in multiple steps

        mesh_rep: has shape (B, N_mesh, d_h)
        Returns mesh_rep: (B, N_mesh, d_h)
        """
        # Embed m2m here first
        return NotImplementedError("process_step not implemented")
