import torchvision
import torch
from collections.abc import Sequence
from torchvision.transforms import functional as F
import numpy as np
from skimage.transform import AffineTransform
import torchvision.transforms.functional as transform
from torch import Tensor
from typing import Tuple, List, Optional
from PIL import Image
import numbers


def _setup_angle(x, name, req_sizes=(2,)):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError(f"If {name} is a single number, it must be positive.")
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError(f"{name} should be a sequence of length {msg}.")
    if len(x) not in req_sizes:
        raise ValueError(f"{name} should be sequence of length {msg}.")


def get_image_size(img: Tensor) -> List[int]:

    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(get_image_size)
    if isinstance(img, torch.Tensor):
        return F_t.get_image_size(img)

    return F_pil.get_image_size(img)


class RandomAffine(torch.nn.Module):
    def __init__(
            self,
            degrees,
            translate=None,
            scale=None,
            shear=None,
            patch_translate=None,
            interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR,
            fill=0,
            fillcolor=None,
            resample=None,
            center=None,
    ):
        super().__init__()
        torchvision.utils._log_api_usage_once(self)
        if resample is not None:
            warnings.warn(
                "The parameter 'resample' is deprecated since 0.12 and will be removed in 0.14. "
                "Please use 'interpolation' instead."
            )
            interpolation = _interpolation_modes_from_int(resample)

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        if fillcolor is not None:
            warnings.warn(
                "The parameter 'fillcolor' is deprecated since 0.12 and will be removed in 0.14. "
                "Please use 'fill' instead."
            )
            fill = fillcolor

        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2,))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2,))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if patch_translate is not None:
            _check_sequence_input(patch_translate, "translate", req_sizes=(2,))
            for k in patch_translate:
                for t in k:
                    if not (0.0 <= t <= 1.0):
                        raise ValueError("translation values should be between 0 and 1")
        self.patch_translate = patch_translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2,))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.resample = self.interpolation = interpolation

        if fill is None:
            fill = 0
        elif not isinstance(fill, (Sequence, numbers.Number)):
            raise TypeError("Fill should be either a sequence or a number.")

        self.fillcolor = self.fill = fill

        if center is not None:
            _check_sequence_input(center, "center", req_sizes=(2,))

        self.center = center

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int],
            patch_translate: Optional[List[float]],
    ) -> Tuple[float, Tuple[int, int], Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)
        if patch_translate is not None:
            min_dx = float(patch_translate[0][0] * img_size[0])
            min_dy = float(patch_translate[1][0] * img_size[1])
            max_dx = float(patch_translate[0][1] * img_size[0])
            max_dy = float(patch_translate[1][1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(min_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(min_dy, max_dy).item()))
            patch_translations = (tx, ty)
        else:
            patch_translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear, patch_translations

    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """
        fill = self.fill

        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            else:
                fill = [float(f) for f in fill]

        img_size = F.get_image_size(img)

        angle, translations, scale, shear, patch_location = self.get_params(self.degrees, self.translate, self.scale,
                                                                            self.shear, img_size, self.patch_translate)
        matrix = torch.eye(3)
        # print(angle, translations, scale, shear, patch_location)

        matrix = torchvision.transforms.functional._get_inverse_affine_matrix([0, 0], angle, translations, scale,
                                                                              shear, )

        return F.affine(img, angle, translations, scale, shear, interpolation=self.interpolation, fill=fill,
                        center=patch_location), matrix, patch_location

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(degrees={self.degrees}"
        s += f", translate={self.translate}" if self.translate is not None else ""
        s += f", scale={self.scale}" if self.scale is not None else ""
        s += f", shear={self.shear}" if self.shear is not None else ""
        s += f", interpolation={self.interpolation.value}" if self.interpolation != InterpolationMode.NEAREST else ""
        s += f", fill={self.fill}" if self.fill != 0 else ""
        s += f", center={self.center}" if self.center is not None else ""
        s += f", patch_translate={self.patch_translate}" if self.patch_translate is not None else ""

        s += ")"

        return s

def translation_transform(tx,ty):
    transform=np.eye(3)
    transform[0,2]=tx
    transform[1,2]=ty
    return transform



def get_random_affine_patches(img, patch_radius = 128, degrees=60, translate=(.05, .05), scale=(.8, 1.2), shear=(.1, .1), patch_translate=((.3, 0.7), (.3, 0.7)) ):
#     print(np.array(img).shape)
    random_affine = RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear, patch_translate=patch_translate)
    transformed_img, affine_matrix, patch_location = random_affine(img)
#     print(np.array(transformed_img).shape)
    
    tr_mat = torch.Tensor(affine_matrix + [0, 0, 1]).reshape(3, 3)
    final_tr_mat = translation_transform(patch_radius, patch_radius) @ tr_mat.numpy() @ np.linalg.inv(
        translation_transform(patch_radius, patch_radius))
    n, m = patch_location
    affine_tr = AffineTransform(matrix=final_tr_mat)
    img = transform.to_tensor(img)
#     print(np.array(img).shape)
    
    transformed_img = transform.to_tensor( transformed_img)
    patch1 = img[:, m - patch_radius:m + patch_radius, n - patch_radius:n + patch_radius]
    patch2 = transformed_img[ :, m - patch_radius:m + patch_radius,
             n - patch_radius:n + patch_radius]
    
    return patch1, patch2, affine_tr

