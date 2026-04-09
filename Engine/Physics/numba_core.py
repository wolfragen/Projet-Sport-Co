# -*- coding: utf-8 -*-
"""
Numba-compiled physics kernel for 2D football simulation.
All functions are @njit — zero Python objects in the hot path.

Convention: entity index 0 = ball (circle), indices 1..N = players (OBB).
shape_type: 0 = circle, 1 = OBB (oriented bounding box).
"""

import numba
from numba import njit
from math import sqrt, cos, sin


# ============================================================
# Geometry utilities
# ============================================================

@njit(cache=True)
def _closest_point_on_segment(px, py, ax, ay, bx, by):
    """Closest point on segment AB to point P. Returns (cx, cy)."""
    abx = bx - ax
    aby = by - ay
    ab_sq = abx * abx + aby * aby
    if ab_sq < 1e-12:
        return ax, ay
    t = ((px - ax) * abx + (py - ay) * aby) / ab_sq
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return ax + t * abx, ay + t * aby


@njit(cache=True)
def _obb_vertices(px, py, hw, hh, angle):
    """4 OBB vertices as flat tuple (x0,y0, x1,y1, x2,y2, x3,y3)."""
    ca = cos(angle)
    sa = sin(angle)
    # local corners: (-hw,-hh), (hw,-hh), (hw,hh), (-hw,hh)
    x0 = px + (-hw) * ca - (-hh) * sa
    y0 = py + (-hw) * sa + (-hh) * ca
    x1 = px + ( hw) * ca - (-hh) * sa
    y1 = py + ( hw) * sa + (-hh) * ca
    x2 = px + ( hw) * ca - ( hh) * sa
    y2 = py + ( hw) * sa + ( hh) * ca
    x3 = px + (-hw) * ca - ( hh) * sa
    y3 = py + (-hw) * sa + ( hh) * ca
    return x0, y0, x1, y1, x2, y2, x3, y3


# ============================================================
# Collision response
# ============================================================

@njit(cache=True)
def _resolve_dynamic(
    pos_x, pos_y, vel_x, vel_y, ang_vel,
    inv_mass, inv_moment,
    i, j,
    depth, nx, ny, cx, cy,
    e, mu
):
    """Impulse-based collision response between two dynamic bodies i and j."""
    # Lever arms
    rax = cx - pos_x[i]
    ray = cy - pos_y[i]
    rbx = cx - pos_x[j]
    rby = cy - pos_y[j]

    # Relative velocity at contact (2D cross: omega x r = (-omega*ry, omega*rx))
    vrx = (vel_x[i] - ang_vel[i] * ray) - (vel_x[j] - ang_vel[j] * rby)
    vry = (vel_y[i] + ang_vel[i] * rax) - (vel_y[j] + ang_vel[j] * rbx)

    # Normal component
    vn = vrx * nx + vry * ny
    if vn > 0.0:
        return  # separating

    # Effective mass (includes angular contribution)
    rn_a = rax * ny - ray * nx  # cross(r_a, n)
    rn_b = rbx * ny - rby * nx
    k = inv_mass[i] + inv_mass[j] + rn_a * rn_a * inv_moment[i] + rn_b * rn_b * inv_moment[j]
    if k < 1e-12:
        return

    # Normal impulse
    jn = -(1.0 + e) * vn / k

    # Apply normal impulse
    vel_x[i] += jn * nx * inv_mass[i]
    vel_y[i] += jn * ny * inv_mass[i]
    ang_vel[i] += rn_a * jn * inv_moment[i]
    vel_x[j] -= jn * nx * inv_mass[j]
    vel_y[j] -= jn * ny * inv_mass[j]
    ang_vel[j] -= rn_b * jn * inv_moment[j]

    # Friction impulse (tangential)
    tx = vrx - vn * nx
    ty = vry - vn * ny
    vt = sqrt(tx * tx + ty * ty)
    if vt > 1e-10:
        tx /= vt
        ty /= vt
        rt_a = rax * ty - ray * tx
        rt_b = rbx * ty - rby * tx
        k_t = inv_mass[i] + inv_mass[j] + rt_a * rt_a * inv_moment[i] + rt_b * rt_b * inv_moment[j]
        if k_t > 1e-12:
            jt = -vt / k_t
            max_f = mu * jn
            if jt < -max_f:
                jt = -max_f
            elif jt > max_f:
                jt = max_f
            vel_x[i] += jt * tx * inv_mass[i]
            vel_y[i] += jt * ty * inv_mass[i]
            ang_vel[i] += rt_a * jt * inv_moment[i]
            vel_x[j] -= jt * tx * inv_mass[j]
            vel_y[j] -= jt * ty * inv_mass[j]
            ang_vel[j] -= rt_b * jt * inv_moment[j]

    # Position correction (Baumgarte)
    SLOP = 0.5
    CORRECTION = 0.4
    total_inv = inv_mass[i] + inv_mass[j]
    if total_inv > 1e-12:
        corr = max(0.0, depth - SLOP) * CORRECTION / total_inv
        pos_x[i] += corr * inv_mass[i] * nx
        pos_y[i] += corr * inv_mass[i] * ny
        pos_x[j] -= corr * inv_mass[j] * nx
        pos_y[j] -= corr * inv_mass[j] * ny


@njit(cache=True)
def _resolve_static(
    pos_x, pos_y, vel_x, vel_y, ang_vel,
    inv_mass_i, inv_moment_i,
    i,
    depth, nx, ny, cx, cy,
    e, mu
):
    """Impulse-based collision response: dynamic body i vs static wall."""
    rax = cx - pos_x[i]
    ray = cy - pos_y[i]

    vrx = vel_x[i] - ang_vel[i] * ray
    vry = vel_y[i] + ang_vel[i] * rax

    vn = vrx * nx + vry * ny
    if vn > 0.0:
        return

    rn_a = rax * ny - ray * nx
    k = inv_mass_i + rn_a * rn_a * inv_moment_i
    if k < 1e-12:
        return

    jn = -(1.0 + e) * vn / k

    vel_x[i] += jn * nx * inv_mass_i
    vel_y[i] += jn * ny * inv_mass_i
    ang_vel[i] += rn_a * jn * inv_moment_i

    # Friction
    tx = vrx - vn * nx
    ty = vry - vn * ny
    vt = sqrt(tx * tx + ty * ty)
    if vt > 1e-10:
        tx /= vt
        ty /= vt
        rt_a = rax * ty - ray * tx
        k_t = inv_mass_i + rt_a * rt_a * inv_moment_i
        if k_t > 1e-12:
            jt = -vt / k_t
            max_f = mu * jn
            if jt < -max_f:
                jt = -max_f
            elif jt > max_f:
                jt = max_f
            vel_x[i] += jt * tx * inv_mass_i
            vel_y[i] += jt * ty * inv_mass_i
            ang_vel[i] += rt_a * jt * inv_moment_i

    # Position correction
    SLOP = 0.5
    CORRECTION = 0.4
    if inv_mass_i > 1e-12:
        corr = max(0.0, depth - SLOP) * CORRECTION / inv_mass_i
        pos_x[i] += corr * inv_mass_i * nx
        pos_y[i] += corr * inv_mass_i * ny


# ============================================================
# Collision detection: Circle vs Capsule (ball vs wall)
# ============================================================

@njit(cache=True)
def _collide_circle_capsule(
    pos_x, pos_y, vel_x, vel_y, ang_vel,
    inv_mass, inv_moment, elasticity, friction, radius,
    i,
    wall_ax, wall_ay, wall_bx, wall_by, wall_radius,
    wall_elasticity, wall_friction, w
):
    cx, cy = pos_x[i], pos_y[i]
    qx, qy = _closest_point_on_segment(cx, cy, wall_ax[w], wall_ay[w], wall_bx[w], wall_by[w])

    dx = cx - qx
    dy = cy - qy
    dist = sqrt(dx * dx + dy * dy)
    pen = radius[i] + wall_radius[w] - dist

    if pen <= 0.0:
        return

    if dist < 1e-10:
        nx, ny = 0.0, 1.0
    else:
        nx = dx / dist
        ny = dy / dist

    contact_x = qx + nx * wall_radius[w]
    contact_y = qy + ny * wall_radius[w]

    e = elasticity[i] * wall_elasticity[w]
    mu = friction[i] * wall_friction[w]

    _resolve_static(pos_x, pos_y, vel_x, vel_y, ang_vel,
                    inv_mass[i], inv_moment[i], i,
                    pen, nx, ny, contact_x, contact_y, e, mu)


# ============================================================
# Collision detection: Circle vs OBB (ball vs player)
# ============================================================

@njit(cache=True)
def _collide_circle_obb(
    pos_x, pos_y, vel_x, vel_y, ang_vel, angle,
    inv_mass, inv_moment, elasticity, friction, radius, half_w, half_h,
    ci, bi
):
    """Circle ci vs OBB bi."""
    ccx, ccy = pos_x[ci], pos_y[ci]
    bpx, bpy = pos_x[bi], pos_y[bi]
    ba = angle[bi]
    bhw, bhh = half_w[bi], half_h[bi]

    # Transform circle center to OBB local frame
    dx = ccx - bpx
    dy = ccy - bpy
    ca = cos(ba)
    sa = sin(ba)
    local_x = dx * ca + dy * sa
    local_y = -dx * sa + dy * ca

    # Clamp to box extents
    clamped_x = max(-bhw, min(local_x, bhw))
    clamped_y = max(-bhh, min(local_y, bhh))

    # Distance from circle center to closest point on box (in local frame)
    diff_x = local_x - clamped_x
    diff_y = local_y - clamped_y
    dist_sq = diff_x * diff_x + diff_y * diff_y
    cr = radius[ci]

    if dist_sq > cr * cr and dist_sq > 1e-10:
        return  # no collision

    if dist_sq < 1e-10:
        # Circle center inside OBB — find axis of minimum penetration
        pen_x = bhw - abs(local_x)
        pen_y = bhh - abs(local_y)
        if pen_x < pen_y:
            depth = pen_x + cr
            if local_x >= 0:
                lnx, lny = 1.0, 0.0
            else:
                lnx, lny = -1.0, 0.0
        else:
            depth = pen_y + cr
            if local_y >= 0:
                lnx, lny = 0.0, 1.0
            else:
                lnx, lny = 0.0, -1.0
        # Contact on box surface in local frame
        local_cx = clamped_x + lnx * 0.0  # on surface
        local_cy = clamped_y + lny * 0.0
        if pen_x < pen_y:
            local_cx = bhw if local_x >= 0 else -bhw
            local_cy = local_y
        else:
            local_cx = local_x
            local_cy = bhh if local_y >= 0 else -bhh
    else:
        dist = sqrt(dist_sq)
        depth = cr - dist
        lnx = diff_x / dist
        lny = diff_y / dist
        local_cx = clamped_x
        local_cy = clamped_y

    # Transform normal and contact back to world frame
    nx = lnx * ca - lny * sa
    ny = lnx * sa + lny * ca
    contact_x = bpx + local_cx * ca - local_cy * sa
    contact_y = bpy + local_cx * sa + local_cy * ca

    e = elasticity[ci] * elasticity[bi]
    mu = friction[ci] * friction[bi]

    _resolve_dynamic(pos_x, pos_y, vel_x, vel_y, ang_vel,
                     inv_mass, inv_moment, ci, bi,
                     depth, nx, ny, contact_x, contact_y, e, mu)


# ============================================================
# Collision detection: OBB vs Capsule (player vs wall)
# ============================================================

@njit(cache=True)
def _collide_obb_capsule(
    pos_x, pos_y, vel_x, vel_y, ang_vel, angle,
    inv_mass, inv_moment, elasticity, friction, half_w, half_h,
    i,
    wall_ax, wall_ay, wall_bx, wall_by, wall_radius,
    wall_elasticity, wall_friction, w
):
    """OBB i vs wall capsule w. Approximate: closest point on segment → circle vs OBB."""
    bpx, bpy = pos_x[i], pos_y[i]
    ba = angle[i]
    bhw, bhh = half_w[i], half_h[i]

    # Closest point on wall segment to OBB center
    qx, qy = _closest_point_on_segment(bpx, bpy, wall_ax[w], wall_ay[w], wall_bx[w], wall_by[w])
    wr = wall_radius[w]

    # Now solve: circle(q, wr) vs OBB(i)
    # Transform circle center (q) to OBB local frame
    dx = qx - bpx
    dy = qy - bpy
    ca = cos(ba)
    sa = sin(ba)
    local_x = dx * ca + dy * sa
    local_y = -dx * sa + dy * ca

    clamped_x = max(-bhw, min(local_x, bhw))
    clamped_y = max(-bhh, min(local_y, bhh))

    diff_x = local_x - clamped_x
    diff_y = local_y - clamped_y
    dist_sq = diff_x * diff_x + diff_y * diff_y

    if dist_sq > wr * wr and dist_sq > 1e-10:
        return

    if dist_sq < 1e-10:
        pen_x = bhw - abs(local_x)
        pen_y = bhh - abs(local_y)
        if pen_x < pen_y:
            depth = pen_x + wr
            if local_x >= 0:
                lnx, lny = 1.0, 0.0
            else:
                lnx, lny = -1.0, 0.0
        else:
            depth = pen_y + wr
            if local_y >= 0:
                lnx, lny = 0.0, 1.0
            else:
                lnx, lny = 0.0, -1.0
    else:
        dist = sqrt(dist_sq)
        depth = wr - dist
        lnx = diff_x / dist
        lny = diff_y / dist

    # Normal points from wall toward OBB → invert (points from "circle" to OBB)
    # but we want normal pointing toward the dynamic body (OBB)
    # Since the "circle" is the wall point and OBB is the body,
    # the normal should point from wall toward OBB = -(lnx, lny) in the circle-vs-obb sense
    # Actually: in circle_obb, normal points from OBB toward circle.
    # Here we want normal pointing from wall toward OBB (the dynamic body).
    # So we negate.
    lnx = -lnx
    lny = -lny

    nx = lnx * ca - lny * sa
    ny = lnx * sa + lny * ca

    contact_x = qx - nx * wr  # contact on wall surface
    contact_y = qy - ny * wr

    e = elasticity[i] * wall_elasticity[w]
    mu = friction[i] * wall_friction[w]

    _resolve_static(pos_x, pos_y, vel_x, vel_y, ang_vel,
                    inv_mass[i], inv_moment[i], i,
                    depth, nx, ny, contact_x, contact_y, e, mu)


# ============================================================
# Collision detection: OBB vs OBB (player vs player) — SAT
# ============================================================

@njit(cache=True)
def _sat_overlap(
    v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y,
    u0x, u0y, u1x, u1y, u2x, u2y, u3x, u3y,
    ax, ay
):
    """Project 2 sets of 4 vertices onto axis (ax, ay). Return overlap (negative = gap)."""
    # Project first set
    d0 = v0x * ax + v0y * ay
    d1 = v1x * ax + v1y * ay
    d2 = v2x * ax + v2y * ay
    d3 = v3x * ax + v3y * ay
    min_a = min(d0, min(d1, min(d2, d3)))
    max_a = max(d0, max(d1, max(d2, d3)))
    # Project second set
    d0 = u0x * ax + u0y * ay
    d1 = u1x * ax + u1y * ay
    d2 = u2x * ax + u2y * ay
    d3 = u3x * ax + u3y * ay
    min_b = min(d0, min(d1, min(d2, d3)))
    max_b = max(d0, max(d1, max(d2, d3)))
    return min(max_a - min_b, max_b - min_a)


@njit(cache=True)
def _collide_obb_obb(
    pos_x, pos_y, vel_x, vel_y, ang_vel, angle,
    inv_mass, inv_moment, elasticity, friction, half_w, half_h,
    i, j
):
    """OBB i vs OBB j using SAT with 4 axes."""
    v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y = _obb_vertices(
        pos_x[i], pos_y[i], half_w[i], half_h[i], angle[i])
    u0x, u0y, u1x, u1y, u2x, u2y, u3x, u3y = _obb_vertices(
        pos_x[j], pos_y[j], half_w[j], half_h[j], angle[j])

    # 4 separating axes: 2 edge normals per box
    ca_i = cos(angle[i])
    sa_i = sin(angle[i])
    ca_j = cos(angle[j])
    sa_j = sin(angle[j])

    axes_x = (ca_i, sa_i, ca_j, sa_j)
    axes_y = (-sa_i, ca_i, -sa_j, ca_j)
    # Actually the normals to the edges of a box rotated by angle are:
    # (cos(a), sin(a)) and (-sin(a), cos(a))
    # which is what we have

    min_overlap = 1e18
    best_nx = 0.0
    best_ny = 0.0

    for k in range(4):
        ax_k = axes_x[k]
        ay_k = axes_y[k]
        overlap = _sat_overlap(
            v0x, v0y, v1x, v1y, v2x, v2y, v3x, v3y,
            u0x, u0y, u1x, u1y, u2x, u2y, u3x, u3y,
            ax_k, ay_k)
        if overlap <= 0.0:
            return  # separating axis found
        if overlap < min_overlap:
            min_overlap = overlap
            best_nx = ax_k
            best_ny = ay_k

    # Ensure normal points from j toward i
    dx = pos_x[i] - pos_x[j]
    dy = pos_y[i] - pos_y[j]
    if dx * best_nx + dy * best_ny < 0.0:
        best_nx = -best_nx
        best_ny = -best_ny

    # Contact point: midpoint of centers (simplified)
    contact_x = (pos_x[i] + pos_x[j]) * 0.5
    contact_y = (pos_y[i] + pos_y[j]) * 0.5

    e = elasticity[i] * elasticity[j]
    mu = friction[i] * friction[j]

    _resolve_dynamic(pos_x, pos_y, vel_x, vel_y, ang_vel,
                     inv_mass, inv_moment, i, j,
                     min_overlap, best_nx, best_ny, contact_x, contact_y, e, mu)


# ============================================================
# Main step kernel
# ============================================================

@njit(cache=True)
def step_kernel(
    pos_x, pos_y, vel_x, vel_y, angle, ang_vel,
    mass, inv_mass, moment, inv_moment,
    elasticity, friction, shape_type, radius, half_w, half_h,
    n_bodies,
    wall_ax, wall_ay, wall_bx, wall_by,
    wall_radius, wall_elasticity, wall_friction, n_walls,
    dt_ms, n_substeps, damping
):
    sub_dt = dt_ms / n_substeps / 1000.0
    damping_factor = damping ** sub_dt

    for _step in range(n_substeps):
        # 1) Damping
        for i in range(n_bodies):
            vel_x[i] *= damping_factor
            vel_y[i] *= damping_factor
            ang_vel[i] *= damping_factor

        # 2) Integration
        for i in range(n_bodies):
            pos_x[i] += vel_x[i] * sub_dt
            pos_y[i] += vel_y[i] * sub_dt
            angle[i] += ang_vel[i] * sub_dt

        # 3) Body vs Wall collisions
        for i in range(n_bodies):
            for w in range(n_walls):
                if shape_type[i] == 0:
                    _collide_circle_capsule(
                        pos_x, pos_y, vel_x, vel_y, ang_vel,
                        inv_mass, inv_moment, elasticity, friction, radius,
                        i,
                        wall_ax, wall_ay, wall_bx, wall_by, wall_radius,
                        wall_elasticity, wall_friction, w)
                else:
                    _collide_obb_capsule(
                        pos_x, pos_y, vel_x, vel_y, ang_vel, angle,
                        inv_mass, inv_moment, elasticity, friction, half_w, half_h,
                        i,
                        wall_ax, wall_ay, wall_bx, wall_by, wall_radius,
                        wall_elasticity, wall_friction, w)

        # 4) Body vs Body collisions
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                st_i = shape_type[i]
                st_j = shape_type[j]
                if st_i == 0 and st_j == 1:
                    _collide_circle_obb(
                        pos_x, pos_y, vel_x, vel_y, ang_vel, angle,
                        inv_mass, inv_moment, elasticity, friction,
                        radius, half_w, half_h, i, j)
                elif st_i == 1 and st_j == 0:
                    _collide_circle_obb(
                        pos_x, pos_y, vel_x, vel_y, ang_vel, angle,
                        inv_mass, inv_moment, elasticity, friction,
                        radius, half_w, half_h, j, i)
                elif st_i == 1 and st_j == 1:
                    _collide_obb_obb(
                        pos_x, pos_y, vel_x, vel_y, ang_vel, angle,
                        inv_mass, inv_moment, elasticity, friction,
                        half_w, half_h, i, j)
