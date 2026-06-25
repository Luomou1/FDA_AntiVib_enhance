from __future__ import annotations

"""生成应用图标。"""

from pathlib import Path

from PIL import Image, ImageDraw


def _rounded_rect(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], radius: int, fill: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill)


def build_icon(size: int) -> Image.Image:
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    scale = size / 256

    def s(value: int) -> int:
        return int(round(value * scale))

    _rounded_rect(draw, (s(18), s(18), s(238), s(238)), s(42), (30, 58, 92, 255))
    _rounded_rect(draw, (s(33), s(34), s(223), s(222)), s(30), (246, 248, 250, 255))

    axis_color = (69, 82, 99, 255)
    draw.line((s(58), s(183), s(203), s(183)), fill=axis_color, width=max(1, s(7)))
    draw.line((s(58), s(63), s(58), s(183)), fill=axis_color, width=max(1, s(7)))

    bars = [
        (s(78), s(137), s(99), s(183), (36, 111, 159, 255)),
        (s(113), s(113), s(134), s(183), (35, 107, 99, 255)),
        (s(148), s(82), s(169), s(183), (29, 111, 159, 255)),
    ]
    for x0, y0, x1, y1, color in bars:
        _rounded_rect(draw, (x0, y0, x1, y1), max(1, s(6)), color)

    points = [(s(75), s(128)), (s(108), s(119)), (s(138), s(96)), (s(173), s(72)), (s(200), s(88))]
    draw.line(points, fill=(184, 74, 45, 255), width=max(1, s(8)), joint="curve")
    for point in points:
        radius = s(7)
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=(184, 74, 45, 255),
        )

    return image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    assets_dir = root / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    sizes = [build_icon(size) for size in (16, 24, 32, 48, 64, 128, 256)]
    sizes[-1].save(assets_dir / "app_icon.ico", sizes=[(image.width, image.height) for image in sizes], append_images=sizes[:-1])


if __name__ == "__main__":
    main()
