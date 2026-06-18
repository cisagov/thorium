import { describe, test, expect } from 'vitest';

// project imports
import { getOrthogonalPath } from './ThemedEdge';
import { HANDLE_CENTER_Y } from './order';

// Parse the absolute points referenced by an SVG path (M/L endpoints and Q control+end) so we can
// reason about the routed corridor. Returns [x, y] pairs in command order.
function pathPoints(d: string): Array<[number, number]> {
  const tokens = d.trim().split(/[\s,]+/);
  const pts: Array<[number, number]> = [];
  let i = 0;
  while (i < tokens.length) {
    const cmd = tokens[i++];
    if (cmd === 'M' || cmd === 'L') {
      pts.push([parseFloat(tokens[i++]), parseFloat(tokens[i++])]);
    } else if (cmd === 'Q') {
      pts.push([parseFloat(tokens[i++]), parseFloat(tokens[i++])]); // control
      pts.push([parseFloat(tokens[i++]), parseFloat(tokens[i++])]); // end
    } else {
      i++;
    }
  }
  return pts;
}

// Every straight run between consecutive points must be axis-aligned (horizontal or vertical).
function straightSegmentsAxisAligned(d: string): boolean {
  const tokens = d.trim().split(/[\s,]+/);
  let cx = 0;
  let cy = 0;
  let i = 0;
  let ok = true;
  while (i < tokens.length) {
    const cmd = tokens[i++];
    if (cmd === 'M') {
      cx = parseFloat(tokens[i++]);
      cy = parseFloat(tokens[i++]);
    } else if (cmd === 'L') {
      const x = parseFloat(tokens[i++]);
      const y = parseFloat(tokens[i++]);
      if (Math.min(Math.abs(x - cx), Math.abs(y - cy)) > 1.5) ok = false;
      cx = x;
      cy = y;
    } else if (cmd === 'Q') {
      i += 2;
      cx = parseFloat(tokens[i++]);
      cy = parseFloat(tokens[i++]);
    } else {
      i++;
    }
  }
  return ok;
}

describe('getOrthogonalPath', () => {
  test('same-row forward is a straight horizontal line', () => {
    const d = getOrthogonalPath(0, HANDLE_CENTER_Y, 300, HANDLE_CENTER_Y);
    expect(d).toBe(`M 0 ${HANDLE_CENTER_Y} L 300 ${HANDLE_CENTER_Y}`);
  });

  test('forward off-row stays orthogonal (right angles only)', () => {
    expect(straightSegmentsAxisAligned(getOrthogonalPath(0, 17, 300, 60, 'target'))).toBe(true);
  });

  describe('backward detour lane points toward the center line', () => {
    // Backward = target at/left of source, so the router must add a C-shape detour lane.
    test('below the center line → lane routes up (toward center)', () => {
      const sy = 90;
      const ty = 90; // both below HANDLE_CENTER_Y
      const d = getOrthogonalPath(300, sy, 40, ty, 'target');
      expect(straightSegmentsAxisAligned(d)).toBe(true);
      const minY = Math.min(...pathPoints(d).map((p) => p[1]));
      // lane sits above the nodes (smaller y) → legs point up toward the spine
      expect(minY).toBeLessThan(Math.min(sy, ty));
    });

    test('above the center line → lane flips down (toward center)', () => {
      const sy = -60;
      const ty = -60; // both above HANDLE_CENTER_Y
      const d = getOrthogonalPath(300, sy, 40, ty, 'target');
      expect(straightSegmentsAxisAligned(d)).toBe(true);
      const maxY = Math.max(...pathPoints(d).map((p) => p[1]));
      // lane sits below the nodes (larger y) → legs point down toward the spine
      expect(maxY).toBeGreaterThan(Math.max(sy, ty));
    });
  });
});
