import { test, expect } from '@playwright/test';
import { loginViaUI, snapshot } from './helpers';

const OUT = 'e2e/screenshots';

// Instrument the document for ~2.5s and report how often the page-level
// scrollbar appears/disappears (i.e. the root client size oscillates) and
// how often the tooltip popper repositions, while a given trigger is hovered.
async function measure(page: import('@playwright/test').Page, label: string) {
  return await page.evaluate(async (lbl) => {
    const root = document.scrollingElement || document.documentElement;
    const samples: {
      t: number;
      clientW: number;
      clientH: number;
      scrollW: number;
      scrollH: number;
      vOver: boolean; // vertical overflow present
      hOver: boolean; // horizontal overflow present
      tipRect: string;
    }[] = [];
    const start = performance.now();
    return await new Promise<{
      label: string;
      durationMs: number;
      frames: number;
      vToggles: number;
      hToggles: number;
      clientWvalues: number[];
      clientHvalues: number[];
      tipMoves: number;
    }>((resolve) => {
      function tick() {
        const tip = document.querySelector('.tooltip');
        const r = tip ? (tip as HTMLElement).getBoundingClientRect() : null;
        samples.push({
          t: performance.now() - start,
          clientW: root.clientWidth,
          clientH: root.clientHeight,
          scrollW: root.scrollWidth,
          scrollH: root.scrollHeight,
          vOver: root.scrollHeight > root.clientHeight,
          hOver: root.scrollWidth > root.clientWidth,
          tipRect: r ? `${Math.round(r.left)},${Math.round(r.top)},${Math.round(r.width)},${Math.round(r.height)}` : 'none',
        });
        if (performance.now() - start < 2500) {
          requestAnimationFrame(tick);
        } else {
          let vToggles = 0;
          let hToggles = 0;
          let tipMoves = 0;
          for (let i = 1; i < samples.length; i++) {
            if (samples[i].vOver !== samples[i - 1].vOver) vToggles++;
            if (samples[i].hOver !== samples[i - 1].hOver) hToggles++;
            if (samples[i].tipRect !== samples[i - 1].tipRect) tipMoves++;
          }
          resolve({
            label: lbl,
            durationMs: Math.round(samples[samples.length - 1].t),
            frames: samples.length,
            vToggles,
            hToggles,
            clientWvalues: [...new Set(samples.map((s) => s.clientW))],
            clientHvalues: [...new Set(samples.map((s) => s.clientH))],
            tipMoves,
          });
        }
      }
      requestAnimationFrame(tick);
    });
  }, label);
}

test('pipeline create tooltip scrollbar jiggle', async ({ page }) => {
  page.setViewportSize({ width: 1280, height: 900 });
  await loginViaUI(page);
  await page.goto('/create/pipeline');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1000);

  // Baseline: nothing hovered.
  const baseline = await measure(page, 'baseline-no-hover');
  console.log('BASELINE', JSON.stringify(baseline));

  // Hover the "Order" heading which is wrapped in OverlayTipRight.
  const order = page.getByText('Order', { exact: true }).first();
  await order.scrollIntoViewIfNeeded();
  await order.hover();
  await page.waitForTimeout(300);
  await snapshot(page, OUT, 'jiggle-order-hover');
  const orderHover = await measure(page, 'order-hover');
  console.log('ORDER_HOVER', JSON.stringify(orderHover));

  // Move away, then hover "Triggers".
  await page.mouse.move(5, 5);
  await page.waitForTimeout(300);
  const triggers = page.getByText('Triggers', { exact: true }).first();
  await triggers.scrollIntoViewIfNeeded();
  await triggers.hover();
  await page.waitForTimeout(300);
  const trigHover = await measure(page, 'triggers-hover');
  console.log('TRIGGERS_HOVER', JSON.stringify(trigHover));

  expect(true).toBe(true);
});
