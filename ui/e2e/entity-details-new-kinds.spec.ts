import { test, expect, Page } from '@playwright/test';
import path from 'path';
import { authenticate, createEntityWithMetadata, deleteEntity, loginViaUI, snapshot, TEST_USER, TEST_PASS } from './helpers';

// Live spec for the config-driven entity DETAILS pages of the entity kinds that landed with the file-entities
// feature: Flag, Incident, CompiledFunction, DecompiledFunction, PeSection, PeImport. Each kind is seeded via
// the API (with metadata) and then deep-linked by id to `<basePath>/:entityID`, where the shared
// createEntityDetailsPage factory renders the common ID / Name / Type / Description / Groups rows plus that
// kind's `EntityMetaInfo` metadata block. The details fetch (`getEntity`) hits the entity by id directly, so
// unlike the browse list it does not depend on search indexing and is stable to assert against. Requires a
// live API (THORIUM_API_URL), like graph.spec.ts.
const SCREENSHOT_DIR = path.join(import.meta.dirname, 'screenshots');

// unique suffix so repeated runs / parallel workers don't collide on entity names
const RUN = Date.now();

test.describe('Entity Details — new kinds', () => {
  let token: string;
  // ids created in beforeAll, deleted in afterAll (best effort)
  const created: string[] = [];
  const ids = {
    flag: '',
    incident: '',
    compiled: '',
    decompiled: '',
    peSection: '',
    peImport: '',
  };
  const names = {
    flag: `DetailsFlag-${RUN}`,
    incident: `DetailsIncident-${RUN}`,
    compiled: `DetailsCompiled-${RUN}`,
    decompiled: `DetailsDecompiled-${RUN}`,
    // PE kinds carry the section/library name in the entity name itself
    peSection: `.textseg-${RUN}`,
    peImport: `kernel32-${RUN}.dll`,
  };

  test.beforeAll(async () => {
    token = await authenticate(TEST_USER, TEST_PASS);

    // Flag: scalar suspicion + enum confidence + reasoning, plus optional content
    ids.flag = await createEntityWithMetadata(
      token,
      names.flag,
      'Flag',
      ['system'],
      [
        { key: 'metadata[suspicion]', value: 7 },
        { key: 'metadata[confidence]', value: 'Likely' },
        { key: 'metadata[reasoning]', value: 'seeded-flag-reasoning' },
        { key: 'metadata[content]', value: 'seeded-flag-content' },
      ],
    );

    // Incident: optional cover_term + several list fields
    ids.incident = await createEntityWithMetadata(
      token,
      names.incident,
      'Incident',
      ['system'],
      [
        { key: 'metadata[cover_term]', value: 'seeded-cover-term' },
        { key: 'metadata[mission_teams][]', value: 'seeded-team-alpha' },
        { key: 'metadata[networks][]', value: 'seeded-net-10' },
        { key: 'metadata[machines][]', value: 'seeded-host-01' },
        { key: 'metadata[locations][]', value: 'seeded-site-hq' },
      ],
    );

    // CompiledFunction: numeric address (rendered as hex) + JSON-encoded disassembly instructions
    ids.compiled = await createEntityWithMetadata(
      token,
      names.compiled,
      'CompiledFunction',
      ['system'],
      [
        { key: 'metadata[function_address]', value: 0x401000 },
        { key: 'metadata[disassembly][]', value: JSON.stringify({ address: 0x401000, instruction: 'push rbp' }) },
        { key: 'metadata[disassembly][]', value: JSON.stringify({ address: 0x401001, instruction: 'mov rbp, rsp' }) },
      ],
    );

    // DecompiledFunction: numeric address + tools list + decompiled source content
    ids.decompiled = await createEntityWithMetadata(
      token,
      names.decompiled,
      'DecompiledFunction',
      ['system'],
      [
        { key: 'metadata[function_address]', value: 0x402000 },
        { key: 'metadata[tools][]', value: 'seeded-ghidra' },
        { key: 'metadata[decompilation_content]', value: 'int seeded_decomp(void) { return 42; }' },
      ],
    );

    // PeSection: optional scalar fields (md5, sizes, entropy); name carries the section name
    ids.peSection = await createEntityWithMetadata(
      token,
      names.peSection,
      'PeSection',
      ['system'],
      [
        { key: 'metadata[md5]', value: 'd41d8cd98f00b204e9800998ecf8427e' },
        { key: 'metadata[raw_size]', value: 4096 },
        { key: 'metadata[virtual_size]', value: 8192 },
        { key: 'metadata[entropy]', value: 6.5 },
      ],
    );

    // PeImport: functions list; name carries the imported library name
    ids.peImport = await createEntityWithMetadata(
      token,
      names.peImport,
      'PeImport',
      ['system'],
      [
        { key: 'metadata[functions][]', value: 'CreateFileA' },
        { key: 'metadata[functions][]', value: 'WriteFile' },
      ],
    );

    created.push(ids.flag, ids.incident, ids.compiled, ids.decompiled, ids.peSection, ids.peImport);
  });

  test.afterAll(async () => {
    for (const id of created) {
      if (id) await deleteEntity(token, id).catch(() => {});
    }
  });

  // Deep-link to a details page by id and wait for the shared factory's Type row to render (which only
  // appears once `getEntity` resolves and the real entity — not the blank default — is in state).
  const openDetails = async (page: Page, basePath: string, id: string, typeLabel: string) => {
    await loginViaUI(page);
    await page.goto(`${basePath}/${id}`);
    await page.waitForLoadState('networkidle');
    // the Type row shows the human label; it distinguishes the loaded entity from the blank placeholder
    await expect(page.getByText('Type', { exact: true })).toBeVisible({ timeout: 30000 });
    await expect(page.getByText(typeLabel, { exact: true }).first()).toBeVisible({ timeout: 30000 });
  };

  test('Flag details renders suspicion, confidence, reasoning, and content', async ({ page }) => {
    await openDetails(page, '/flag', ids.flag, 'Flag');
    // the common header shows the entity id + name
    await expect(page.getByText(ids.flag, { exact: false })).toBeVisible();
    await expect(page.getByText(names.flag, { exact: false }).first()).toBeVisible();
    // FlagMetaInfo rows: labels + seeded values
    await expect(page.getByText('Suspicion', { exact: true })).toBeVisible();
    await expect(page.getByText('7', { exact: true })).toBeVisible();
    await expect(page.getByText('Confidence', { exact: true })).toBeVisible();
    await expect(page.getByText('Likely', { exact: true })).toBeVisible();
    await expect(page.getByText('Reasoning', { exact: true })).toBeVisible();
    await expect(page.getByText('seeded-flag-reasoning')).toBeVisible();
    // content is optional and only rendered when set
    await expect(page.getByText('Content', { exact: true })).toBeVisible();
    await expect(page.getByText('seeded-flag-content')).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'entity-details-flag');
  });

  test('Incident details renders the cover term and list fields', async ({ page }) => {
    await openDetails(page, '/incident', ids.incident, 'Incident');
    await expect(page.getByText('Cover Term', { exact: true })).toBeVisible();
    await expect(page.getByText('seeded-cover-term')).toBeVisible();
    // the four list fields render as labeled rows with FieldBadge values
    await expect(page.getByText('Mission Teams', { exact: true })).toBeVisible();
    await expect(page.getByText('seeded-team-alpha')).toBeVisible();
    await expect(page.getByText('Networks', { exact: true })).toBeVisible();
    await expect(page.getByText('seeded-net-10')).toBeVisible();
    await expect(page.getByText('Machines', { exact: true })).toBeVisible();
    await expect(page.getByText('seeded-host-01')).toBeVisible();
    await expect(page.getByText('Locations', { exact: true })).toBeVisible();
    await expect(page.getByText('seeded-site-hq')).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'entity-details-incident');
  });

  test('Compiled Function details renders the hex address and disassembly instructions', async ({ page }) => {
    await openDetails(page, '/function/compiled', ids.compiled, 'Compiled Function');
    // address is rendered as hex via formatAddress
    await expect(page.getByText('Address', { exact: true })).toBeVisible();
    await expect(page.getByText('0x401000', { exact: false }).first()).toBeVisible();
    // the disassembly renders through the read-only code renderer; instruction mnemonics appear
    await expect(page.getByText('Disassembly', { exact: true })).toBeVisible();
    await expect(page.getByText('push rbp', { exact: false })).toBeVisible({ timeout: 15000 });
    await expect(page.getByText('mov rbp, rsp', { exact: false })).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'entity-details-compiled-function');
  });

  test('Decompiled Function details renders address, tools, and decompiled content', async ({ page }) => {
    await openDetails(page, '/function/decompiled', ids.decompiled, 'Decompiled Function');
    await expect(page.getByText('Address', { exact: true })).toBeVisible();
    await expect(page.getByText('0x402000', { exact: false }).first()).toBeVisible();
    await expect(page.getByText('Tools', { exact: true })).toBeVisible();
    await expect(page.getByText('seeded-ghidra')).toBeVisible();
    await expect(page.getByText('Content', { exact: true })).toBeVisible();
    // the decompiled source renders through the shared code renderer
    await expect(page.getByText('seeded_decomp', { exact: false })).toBeVisible({ timeout: 15000 });
    await snapshot(page, SCREENSHOT_DIR, 'entity-details-decompiled-function');
  });

  test('PE Section details renders md5, sizes, and entropy', async ({ page }) => {
    await openDetails(page, '/pe/section', ids.peSection, 'PE Section');
    // the section name is carried by the entity name (rendered in the header)
    await expect(page.getByText(names.peSection, { exact: false }).first()).toBeVisible();
    await expect(page.getByText('MD5', { exact: true })).toBeVisible();
    await expect(page.getByText('d41d8cd98f00b204e9800998ecf8427e')).toBeVisible();
    await expect(page.getByText('Raw Size', { exact: true })).toBeVisible();
    await expect(page.getByText('4096', { exact: true })).toBeVisible();
    await expect(page.getByText('Virtual Size', { exact: true })).toBeVisible();
    await expect(page.getByText('8192', { exact: true })).toBeVisible();
    await expect(page.getByText('Entropy', { exact: true })).toBeVisible();
    await expect(page.getByText('6.5', { exact: true })).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'entity-details-pe-section');
  });

  test('PE Import details renders the imported functions', async ({ page }) => {
    await openDetails(page, '/pe/import', ids.peImport, 'PE Import');
    // the library name is carried by the entity name (rendered in the header)
    await expect(page.getByText(names.peImport, { exact: false }).first()).toBeVisible();
    await expect(page.getByText('Functions', { exact: true })).toBeVisible();
    await expect(page.getByText('CreateFileA')).toBeVisible();
    await expect(page.getByText('WriteFile', { exact: true })).toBeVisible();
    await snapshot(page, SCREENSHOT_DIR, 'entity-details-pe-import');
  });

  test('the details edit toggle switches a metadata block into editable inputs', async ({ page }) => {
    // Exercise the per-kind editable metadata path on Incident (whose IncidentMetaInfo branches on the
    // shared factory's `editing` flag — Flag's block is read-only, so it wouldn't show the change). The
    // edit toggle is an icon-only button (its label is a hover tooltip, not an accessible name), so we
    // target the first .icon-btn in the details button row and confirm its identity via that tooltip.
    await openDetails(page, '/incident', ids.incident, 'Incident');
    // before editing, the cover term is plain read-only text (no input carries it)
    await expect(page.getByText('seeded-cover-term')).toBeVisible();
    const editBtn = page.locator('button.icon-btn').first();
    await editBtn.hover();
    await expect(page.getByText(`Edit "${names.incident}"`)).toBeVisible({ timeout: 10000 });
    await editBtn.click();
    // the Save affordance confirms edit mode is active (its tooltip surfaces on hover)
    const saveBtn = page.locator('button.icon-btn').nth(1);
    await saveBtn.hover();
    await expect(page.getByText('Save pending changes')).toBeVisible({ timeout: 10000 });
    // the cover-term row is now an editable input pre-filled with the seeded value (the input directly
    // follows the "Cover Term" label within its row)
    const coverTermInput = page.getByText('Cover Term', { exact: true }).locator('xpath=following::input[1]');
    await expect(coverTermInput).toBeVisible({ timeout: 10000 });
    await expect(coverTermInput).toHaveValue('seeded-cover-term');
    await snapshot(page, SCREENSHOT_DIR, 'entity-details-incident-editing');
  });
});
