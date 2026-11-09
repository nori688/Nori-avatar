"""Фоллбэк-процедурная графика для отрисовки пиксельных заглушек при отсутствии PNG."""
import pygame


def draw_face_fallback(surface):
    """Рисует пиксельную голову аниме-персонажа без черт лица."""
    art = pygame.Surface((80, 90), pygame.SRCALPHA)

    outline = (45, 30, 55)
    hair_dark = (58, 40, 78)
    hair = (91, 61, 116)
    hair_light = (132, 86, 145)
    skin_shadow = (219, 153, 145)
    skin = (250, 204, 181)
    skin_light = (255, 224, 204)

    # Задняя масса волос
    pygame.draw.polygon(
        art,
        outline,
        [(13, 28), (18, 13), (29, 5), (50, 4), (63, 12),
         (70, 28), (69, 68), (61, 86), (51, 78), (29, 79),
         (19, 87), (11, 67)],
    )
    pygame.draw.polygon(
        art,
        hair_dark,
        [(16, 29), (20, 15), (30, 8), (49, 7), (61, 14),
         (67, 29), (66, 65), (60, 80), (53, 73), (27, 74),
         (20, 81), (14, 64)],
    )
    pygame.draw.polygon(art, hair, [(17, 27), (22, 15), (34, 9), (27, 71), (19, 78)])
    pygame.draw.polygon(art, hair, [(47, 9), (60, 16), (65, 29), (61, 78), (53, 71)])
    pygame.draw.polygon(art, hair_light, [(25, 13), (30, 10), (24, 60), (20, 66)])
    pygame.draw.polygon(art, hair_light, [(53, 12), (57, 15), (61, 57), (58, 63)])

    # Шея
    pygame.draw.rect(art, outline, (33, 69, 15, 17))
    pygame.draw.rect(art, skin_shadow, (35, 68, 11, 16))
    pygame.draw.rect(art, skin, (36, 68, 9, 14))

    # Уши и форма лица
    pygame.draw.rect(art, outline, (13, 38, 8, 18))
    pygame.draw.rect(art, outline, (59, 38, 8, 18))
    pygame.draw.rect(art, skin_shadow, (15, 40, 6, 14))
    pygame.draw.rect(art, skin_shadow, (59, 40, 6, 14))
    pygame.draw.polygon(
        art,
        outline,
        [(22, 19), (56, 19), (62, 28), (62, 55), (56, 68),
         (47, 77), (34, 77), (24, 69), (18, 56), (18, 29)],
    )
    pygame.draw.polygon(
        art,
        skin,
        [(24, 21), (55, 21), (59, 29), (59, 54), (53, 66),
         (46, 73), (35, 73), (27, 66), (21, 54), (21, 30)],
    )
    pygame.draw.polygon(art, skin_light, [(25, 27), (31, 22), (27, 61), (32, 69), (25, 63)])

    # Румянец и нос
    pygame.draw.rect(art, (239, 143, 151), (23, 52, 7, 2))
    pygame.draw.rect(art, (239, 143, 151), (51, 52, 7, 2))
    pygame.draw.rect(art, skin_shadow, (39, 49, 3, 5))
    pygame.draw.rect(art, skin_light, (39, 49, 1, 3))

    # Чёлка
    pygame.draw.polygon(
        art,
        hair,
        [(20, 25), (23, 16), (32, 10), (39, 9), (36, 29),
         (32, 24), (29, 33), (25, 26), (22, 35)],
    )
    pygame.draw.polygon(
        art,
        hair_dark,
        [(38, 9), (50, 10), (58, 18), (60, 28), (55, 35),
         (52, 25), (48, 32), (45, 23), (39, 29)],
    )
    pygame.draw.rect(art, hair_light, (28, 13, 3, 10))
    pygame.draw.rect(art, hair_light, (49, 13, 3, 8))

    pygame.transform.scale(art, surface.get_size(), surface)


def draw_brow_fallback(surface):
    """Рисует ступенчатую пиксельную бровь."""
    art = pygame.Surface((32, 6), pygame.SRCALPHA)
    color = (69, 42, 66)
    pygame.draw.rect(art, color, (3, 3, 8, 2))
    pygame.draw.rect(art, color, (10, 2, 10, 2))
    pygame.draw.rect(art, color, (19, 1, 9, 2))
    pygame.transform.scale(art, surface.get_size(), surface)


def draw_eye_fallback(surface):
    """Рисует крупный пиксельный аниме-глаз."""
    art = pygame.Surface((34, 22), pygame.SRCALPHA)
    outline = (48, 34, 58)
    white = (255, 246, 244)
    iris_dark = (45, 80, 125)
    iris = (76, 155, 198)

    pygame.draw.polygon(art, outline, [(2, 11), (7, 6), (14, 4), (25, 5), (32, 10), (26, 15), (9, 16)])
    pygame.draw.polygon(art, white, [(5, 11), (9, 8), (15, 6), (24, 7), (29, 10), (25, 13), (10, 14)])
    pygame.draw.rect(art, iris_dark, (14, 5, 10, 11))
    pygame.draw.rect(art, iris, (16, 6, 6, 9))
    pygame.draw.rect(art, (25, 32, 48), (18, 8, 5, 7))
    pygame.draw.rect(art, (245, 250, 255), (17, 7, 2, 3))
    pygame.draw.rect(art, (176, 226, 243), (20, 13, 2, 2))
    pygame.draw.rect(art, outline, (7, 5, 18, 2))
    pygame.draw.rect(art, outline, (3, 8, 7, 2))
    pygame.draw.rect(art, outline, (27, 7, 5, 2))
    pygame.draw.rect(art, outline, (30, 5, 3, 2))
    pygame.transform.scale(art, surface.get_size(), surface)


def draw_mouth_fallback(surface):
    """Рисует рот."""
    art = pygame.Surface((28, 15), pygame.SRCALPHA)
    outline = (91, 39, 63)
    inside = (126, 42, 67)
    tongue = (232, 112, 127)
    pygame.draw.polygon(art, outline, [(3, 5), (8, 2), (20, 2), (25, 5), (22, 12), (7, 12)])
    pygame.draw.polygon(art, inside, [(6, 6), (10, 4), (19, 4), (22, 6), (20, 10), (8, 10)])
    pygame.draw.rect(art, tongue, (9, 9, 11, 2))
    pygame.draw.rect(art, (255, 191, 190), (11, 9, 6, 1))
    pygame.transform.scale(art, surface.get_size(), surface)


def draw_reference_face_fallback(surface):
    """База головы в палитре референса."""
    art = pygame.Surface((80, 90), pygame.SRCALPHA)
    black = (10, 12, 18)
    hair_shadow = (28, 33, 54)
    hair = (48, 55, 86)
    hair_light = (68, 77, 112)
    skin_shadow = (224, 184, 184)
    skin = (248, 235, 233)
    skin_light = (255, 247, 244)
    blush = (246, 150, 166)

    pygame.draw.polygon(
        art,
        black,
        [(10, 31), (14, 18), (23, 8), (34, 3), (52, 4),
         (65, 14), (71, 29), (71, 65), (63, 85), (54, 78),
         (49, 87), (31, 80), (20, 87), (9, 66)],
    )
    pygame.draw.polygon(
        art,
        hair_shadow,
        [(13, 31), (17, 19), (25, 11), (35, 6), (51, 7),
         (62, 16), (68, 30), (68, 64), (62, 79), (55, 72),
         (49, 81), (31, 74), (21, 81), (13, 64)],
    )
    pygame.draw.polygon(art, hair, [(14, 29), (19, 17), (29, 9), (25, 72), (18, 78), (13, 62)])
    pygame.draw.polygon(art, hair, [(48, 7), (61, 17), (67, 31), (66, 63), (60, 78), (53, 71)])
    pygame.draw.polygon(art, hair_light, [(20, 20), (25, 14), (21, 62), (17, 66)])
    pygame.draw.polygon(art, hair_light, [(57, 16), (61, 22), (64, 60), (61, 64)])

    pygame.draw.rect(art, black, (13, 39, 8, 18))
    pygame.draw.rect(art, black, (59, 39, 8, 18))
    pygame.draw.rect(art, skin_shadow, (15, 41, 6, 14))
    pygame.draw.rect(art, skin_shadow, (59, 41, 6, 14))
    pygame.draw.polygon(
        art,
        black,
        [(21, 25), (57, 25), (63, 36), (62, 58), (56, 70),
         (47, 79), (34, 79), (25, 71), (18, 59), (17, 37)],
    )
    pygame.draw.polygon(
        art,
        skin,
        [(23, 28), (55, 28), (59, 37), (59, 57), (53, 68),
         (46, 75), (35, 75), (28, 68), (21, 57), (20, 38)],
    )
    pygame.draw.polygon(art, skin_light, [(24, 33), (28, 29), (25, 62), (31, 70), (25, 66)])
    pygame.draw.rect(art, blush, (21, 52, 9, 2))
    pygame.draw.rect(art, blush, (51, 52, 8, 2))
    pygame.draw.rect(art, skin_shadow, (39, 49, 3, 5))
    pygame.draw.rect(art, skin_light, (39, 49, 1, 3))

    pygame.transform.scale(art, surface.get_size(), surface)


def draw_reference_hair_front_fallback(surface):
    """Передняя чёлка."""
    art = pygame.Surface((80, 90), pygame.SRCALPHA)
    black = (10, 12, 18)
    hair_shadow = (28, 33, 54)
    hair = (48, 55, 86)
    hair_light = (68, 77, 112)
    pink = (244, 149, 167)

    pygame.draw.polygon(
        art,
        black,
        [(15, 31), (17, 19), (27, 9), (38, 5), (53, 7),
         (63, 17), (66, 30), (60, 39), (56, 32), (52, 43),
         (47, 34), (42, 47), (36, 34), (31, 43), (27, 32),
         (22, 39)],
    )
    pygame.draw.polygon(
        art,
        hair,
        [(18, 30), (20, 20), (28, 12), (38, 8), (51, 10),
         (60, 18), (63, 29), (59, 34), (55, 27), (51, 38),
         (47, 28), (42, 41), (37, 28), (32, 38), (28, 27),
         (23, 34)],
    )
    pygame.draw.polygon(art, hair_shadow, [(39, 8), (49, 9), (48, 37), (43, 46), (39, 37)])
    pygame.draw.rect(art, hair_light, (25, 14, 3, 12))
    pygame.draw.rect(art, hair_light, (53, 14, 3, 10))
    pygame.draw.rect(art, black, (18, 33, 13, 7))
    pygame.draw.rect(art, pink, (19, 34, 5, 2))
    pygame.draw.rect(art, pink, (23, 37, 6, 2))
    pygame.draw.rect(art, pink, (55, 35, 6, 2))

    pygame.transform.scale(art, surface.get_size(), surface)


def draw_reference_brow_fallback(surface):
    art = pygame.Surface((30, 6), pygame.SRCALPHA)
    color = (20, 23, 36)
    pygame.draw.rect(art, color, (4, 3, 8, 2))
    pygame.draw.rect(art, color, (11, 2, 9, 2))
    pygame.draw.rect(art, color, (19, 1, 7, 2))
    pygame.transform.scale(art, surface.get_size(), surface)


def draw_reference_eye_fallback(surface):
    art = pygame.Surface((36, 20), pygame.SRCALPHA)
    black = (10, 12, 18)
    white = (250, 244, 243)
    cyan_dark = (61, 127, 143)
    cyan = (128, 206, 216)

    pygame.draw.polygon(art, black, [(1, 10), (7, 6), (16, 5), (27, 6), (35, 10), (29, 15), (8, 15)])
    pygame.draw.polygon(art, white, [(5, 10), (9, 8), (17, 7), (26, 8), (31, 10), (27, 13), (9, 13)])
    pygame.draw.rect(art, cyan_dark, (15, 7, 10, 8))
    pygame.draw.rect(art, cyan, (17, 8, 6, 6))
    pygame.draw.rect(art, black, (19, 9, 4, 6))
    pygame.draw.rect(art, white, (18, 8, 2, 2))
    pygame.draw.rect(art, black, (6, 5, 22, 3))
    pygame.draw.rect(art, black, (2, 7, 7, 3))
    pygame.draw.rect(art, black, (28, 6, 6, 3))
    pygame.transform.scale(art, surface.get_size(), surface)


def draw_reference_mouth_fallback(surface):
    art = pygame.Surface((28, 16), pygame.SRCALPHA)
    black = (10, 12, 18)
    inside = (59, 27, 43)
    pink_dark = (204, 74, 105)
    pink = (247, 116, 144)
    pygame.draw.polygon(art, black, [(2, 4), (8, 2), (21, 2), (26, 5), (23, 13), (6, 13)])
    pygame.draw.polygon(art, inside, [(6, 5), (10, 4), (20, 4), (23, 6), (20, 11), (8, 11)])
    pygame.draw.rect(art, pink_dark, (8, 8, 13, 4))
    pygame.draw.rect(art, pink, (10, 8, 9, 2))
    pygame.transform.scale(art, surface.get_size(), surface)


def draw_reference_body_fallback(surface):
    art = pygame.Surface((104, 72), pygame.SRCALPHA)
    black = (9, 11, 17)
    jacket = (38, 39, 43)
    jacket_light = (68, 69, 69)
    blue_dark = (28, 66, 119)
    blue = (40, 91, 157)
    white = (245, 245, 241)
    cyan = (104, 193, 207)

    pygame.draw.polygon(art, black, [(31, 5), (73, 5), (88, 17), (98, 43), (104, 72), (0, 72), (6, 43), (17, 18)])
    pygame.draw.polygon(art, jacket, [(32, 9), (72, 9), (84, 19), (88, 72), (15, 72), (19, 20)])
    pygame.draw.polygon(art, blue_dark, [(73, 12), (86, 19), (97, 46), (101, 72), (78, 72), (73, 35)])
    pygame.draw.polygon(art, blue, [(83, 25), (91, 34), (98, 61), (97, 72), (86, 72)])
    pygame.draw.polygon(art, jacket_light, [(21, 23), (33, 10), (42, 25), (34, 38)])
    pygame.draw.polygon(art, white, [(36, 8), (50, 17), (42, 30), (31, 16)])
    pygame.draw.polygon(art, white, [(68, 8), (53, 17), (62, 30), (75, 16)])
    pygame.draw.rect(art, white, (43, 25, 4, 32))
    pygame.draw.rect(art, white, (58, 25, 4, 32))
    pygame.draw.rect(art, black, (45, 56, 15, 5))
    pygame.draw.rect(art, cyan, (48, 19, 9, 4))
    pygame.draw.rect(art, (54, 111, 129), (50, 23, 5, 3))

    pygame.transform.scale(art, surface.get_size(), surface)
