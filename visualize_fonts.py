import argparse
import os

from PIL import ImageFont

from hwr_self_train.word_samplers import UniformSampler
from hwr_self_train.data_generator import SimpleRandomWordGenerator


def generate_images(args):
    fonts_dir = args.fonts_dir
    output_dir = args.output_dir
    text = args.text
    font_size = args.font_size
    max_fonts = args.max_fonts

    sampler = UniformSampler([])
    gen = SimpleRandomWordGenerator(sampler, fonts_dir)

    os.makedirs(output_dir, exist_ok=True)

    for i, font_name in enumerate(os.listdir(fonts_dir)):
        if max_fonts and i >= max_fonts:
            break

        if font_name.endswith('.otf') or font_name.endswith('.ttf'):
            font_path = os.path.join(fonts_dir, font_name)
            font = ImageFont.truetype(font_path, size=font_size)
            save_path = os.path.join(output_dir, f"{font_name}.png")
            gen.create_image(text, font, size=font_size, stroke_width=0).save(save_path)
        else:
            print(f"Ignoring file with wrong extension: {font_name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create text images using every font in a given fonts directory'
    )
    parser.add_argument('fonts_dir', type=str,
                        help='Path to the directory containing files with .ttf or .otf extension ')
    parser.add_argument('output_dir', type=str,
                        help='Output directory that will contain generated images')

    parser.add_argument('--text', type=str, default='The quick brown fox jumps over the lazy dog',
                        help='Specifies a text to write using each font')

    parser.add_argument('--font-size', type=int, default=64,
                        help='Font size to use with each font')

    parser.add_argument('--max-fonts', type=int, default=0,
                        help='Number of fonts to visualize. By default, visualize all')

    cli_args = parser.parse_args()
    generate_images(cli_args)
