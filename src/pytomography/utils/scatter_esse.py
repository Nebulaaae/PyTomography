from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from . import get_1d_gaussian_kernel
import numpy as np
import pytomography
import torch
import torch
import torch.nn.functional
from pytomography.utils import get_mu_from_spectrum_interp, get_1d_gaussian_kernel
import os

if TYPE_CHECKING:
    from pytomography.metadata.SPECT import SPECTObjectMeta


class ESSEScatterModel:
    def __init__(
        self, 
        kernels: torch.Tensor, 
        object_meta: "SPECTObjectMeta",
        attenuation_map: torch.Tensor,
        mu_water: torch.Tensor,
        energy: float,
        CGSM: None|int = None
    ):
        """
        Args:
            kernels (torch.Tensor): Kernels de diffusé [3, Z, Y, X]. 
                Le premier index correspond aux 3 composantes de l'Eq. 9.
            object_meta (SPECTObjectMeta): Métadonnées de l'objet SPECT.
        """
        self.object_meta = object_meta
        self.attenuation_map = attenuation_map.to(pytomography.device)
        self.energy = energy
        self.mu_water = mu_water
        self.CGSM = CGSM

        # self.kernels_fft = []
        
        # for i in range(kernels.shape[0]):
        #     k = kernels[i]
            
        #     if self.CGSM and self.CGSM > 1:
        #         k = self.CGSM_collapse(self.CGSM, k)
        #         # todo : l'article propose de diviser les kernels en deux, pour la partie basse résolution et la partie haute résolution. À voir.
            
        #     # Calcul de la FFT sur le noyau
        #     k_shifted = torch.fft.ifftshift(k)
        #     k_fft = torch.fft.rfftn(k_shifted, dim=(-3, -2, -1))
        #     self.kernels_fft.append(k_fft)
        self.fft_shape = [dim * 2 for dim in (self.CGSM_collapse(self.CGSM, kernels[0]).shape if (self.CGSM and self.CGSM > 1) else kernels[0].shape)]
        
        processed_kernels = []
        for i in range(kernels.shape[0]):
            k = kernels[i]
            if self.CGSM and self.CGSM > 1:
                k = self.CGSM_collapse(self.CGSM, k)
            
            k_shifted = torch.fft.ifftshift(k)
            processed_kernels.append(torch.fft.rfftn(k_shifted, s=self.fft_shape, dim=(-3, -2, -1)).to(pytomography.device))
        
        self.kernels_fft = torch.stack(processed_kernels)
        
    def get_relative_electron_density(
        self,
        attenuation_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule la densité électronique relative (rho) à partir de la carte d'atténuation.
        """
        rho = attenuation_map / self.mu_water
        return rho

    def get_depth_map(
        self,
        attenuation_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calcule la carte de profondeur optique à partir de la carte d'atténuation.
        """
        dx = self.object_meta.dx
        if self.CGSM and self.CGSM > 1:
            dx *= self.CGSM
            
        rho = self.get_relative_electron_density(attenuation_map)
        depth_map = torch.cumsum(rho, dim=0) * dx
        
        return depth_map
    
    # todo: vérifier coarse grid (CGSM) pour réduire les temps d'exécution
    def CGSM_collapse(
        self,
        n: int,
        object: torch.Tensor
    ) -> torch.Tensor:
        """
        Réduit la taille de l'objet par un facteur n pour le Coarse-Grid Scatter Modelling.
        """
        if n == 1:
            return object
            
        x = object.unsqueeze(0).unsqueeze(0)
        
        new_size = [dim // n for dim in object.shape]
        
        collapsed = torch.nn.functional.interpolate(
            x, 
            size=new_size, 
            mode='trilinear', 
            # mode='nearest'
            align_corners=True
        )
        
        return collapsed.squeeze(0).squeeze(0)
        
    def CGSM_expand(
    self,
    low_res_scatter: torch.Tensor,
    original_shape: tuple
    ) -> torch.Tensor:
        """
        Réaugmente la résolution du scatter après la convolution sur grille grossière.
        
        Args:
            low_res_scatter: Le scatter calculé en basse résolution.
            original_shape: La forme (Z, Y, X) de l'objet avant le CGSM.
        """
        x = low_res_scatter.unsqueeze(0).unsqueeze(0)
        
        # Interpolation vers la taille d'origine
        expanded = torch.nn.functional.interpolate(
            x, 
            size=original_shape, 
            mode='trilinear',
            # mode='nearest'          # todo : Pas idéal comme mode d'interpolation, utiliser trilinear c'est mieux mais provoque pb sur CPU ?
            align_corners=True
        )
        
        return expanded.squeeze(0).squeeze(0)

    # def prepare_iteration(self, object_3d): #todo: ajouter commentaires
    #     """
        
    #     """
    #     obj = object_3d
    #     if self.CGSM and self.CGSM > 1:
    #         obj = self.CGSM_collapse(self.CGSM, obj)
            
    #     obj_fft = torch.fft.rfftn(obj)
    #     I1 = torch.fft.irfftn(obj_fft * self.kernels_fft[0], s=obj.shape)
    #     I2 = torch.fft.irfftn(obj_fft * self.kernels_fft[1], s=obj.shape)
    #     I3 = torch.fft.irfftn(obj_fft * self.kernels_fft[2], s=obj.shape)
    #     self.convolved_volumes = [I1, I2, I3]

    def prepare_iteration(self, object_3d): #todo: ajouter commentaires
        """
        
        """
        if self.CGSM and self.CGSM > 1:
            object_3d = self.CGSM_collapse(self.CGSM, object_3d)
        # obj_fft = torch.fft.rfftn(object_3d)
        # convolved_fft = obj_fft.unsqueeze(0) * self.kernels_fft
        # self.convolved_volumes = torch.fft.irfftn(convolved_fft, s=object_3d.shape, dim=(-3, -2, -1))
        img_shape = object_3d.shape
        fft_shape = [img_shape[i] * 2 for i in range(len(img_shape))] 
        obj_fft = torch.fft.rfftn(object_3d, s=fft_shape)
        convolved_fft = obj_fft.unsqueeze(0) * self.kernels_fft
        full_conv = torch.fft.irfftn(convolved_fft, s=fft_shape, dim=(-3, -2, -1))
        self.convolved_volumes = full_conv[..., :img_shape[0], :img_shape[1], :img_shape[2]]

    def get_effective_source(self, rho, tau, rotation_transform, angle): #todo: ajouter commentaires
        #todo : regarder la question du zero padding pour éviter les effets de repliement
        """
            
        
        """
        # Rotation des volumes pré-calculés vers l'angle actuel
        I_rot = []
        for vol in self.convolved_volumes:
            rotated_vol = rotation_transform.backward(vol, 270 - angle) 
            if self.CGSM and self.CGSM > 1:
                rotated_vol = self.CGSM_expand(rotated_vol, self.object_meta.padded_shape)
            I_rot.append(rotated_vol)
            
        # Equation 9 Frey et Tsui 1996
        term1 = I_rot[0]
        term2 = I_rot[1] * tau
        term3 = 0.5 * I_rot[2] * (tau ** 2)
        
        return rho * (term1 - term2 + term3)
    
    def apply_adjoint(self, V1: torch.Tensor, V2: torch.Tensor, V3: torch.Tensor) -> torch.Tensor:
        """
    
        """
        if self.CGSM and self.CGSM > 1:
            V1 = self.CGSM_collapse(self.CGSM, V1)
            V2 = self.CGSM_collapse(self.CGSM, V2)
            V3 = self.CGSM_collapse(self.CGSM, V3)
        
        v_shape = V1.shape
        
        V1_fft = torch.fft.rfftn(V1, s=self.fft_shape)
        V2_fft = torch.fft.rfftn(V2, s=self.fft_shape)
        V3_fft = torch.fft.rfftn(V3, s=self.fft_shape)
        
        S1 = torch.fft.irfftn(V1_fft * torch.conj(self.kernels_fft[0]), s=self.fft_shape)
        S2 = torch.fft.irfftn(V2_fft * torch.conj(self.kernels_fft[1]), s=self.fft_shape)
        S3 = torch.fft.irfftn(V3_fft * torch.conj(self.kernels_fft[2]), s=self.fft_shape)

        S1 = S1[..., :v_shape[0], :v_shape[1], :v_shape[2]]
        S2 = S2[..., :v_shape[0], :v_shape[1], :v_shape[2]]
        S3 = S3[..., :v_shape[0], :v_shape[1], :v_shape[2]]

        S = S1 - S2 + S3

        if self.CGSM and self.CGSM > 1:
            S = self.CGSM_expand(S, self.object_meta.padded_shape)
        return S
