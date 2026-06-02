import { describe, test, expect } from 'vitest';
import { ImageChecker } from './index';
import { transformVersion } from './schema';
import { removeLine, replaceLine } from '../../test-helpers';
import { FieldValueType, Severity } from '../../types';

const VALID_IMAGE = `group: analysis
name: yara-scanner
image: thorium/yara-scanner:latest
scaler: K8s
timeout: 300
lifetime:
    counter: jobs
    amount: 32
resources:
    cpu: 1000
    memory: 512
display_type: JSON
collect_logs: true
generator: false
dependencies:
    samples:
        location: /tmp/thorium/samples
        strategy: Paths
        naming: Sha256
output_collection:
    handler: Files
    files:
        results: /tmp/thorium/results
description: Scans files with YARA rules`;

const imageChecker = new ImageChecker();

function imageErrors(text: string) {
  return imageChecker.check(text).diagnostics.filter((d) => d.severity === Severity.Error);
}

function imageWarnings(text: string) {
  return imageChecker.check(text).diagnostics.filter((d) => d.severity === Severity.Warning);
}

function imageSuggestions(text: string) {
  return imageChecker.check(text).suggestions;
}

describe('ImageChecker', () => {
  describe('valid image', () => {
    test('produces no errors for valid image request', () => {
      const errs = imageErrors(VALID_IMAGE);
      expect(errs).toHaveLength(0);
    });

    test('produces no warnings for valid image request', () => {
      const warns = imageWarnings(VALID_IMAGE);
      expect(warns).toHaveLength(0);
    });

    test('root-level suggestions are only for missing known fields', () => {
      const s = imageSuggestions(VALID_IMAGE);
      const rootSuggestions = s.filter((sg) => !sg.field.includes('.'));
      for (const sg of rootSuggestions) {
        expect(VALID_IMAGE).not.toContain(`${sg.field}:`);
      }
    });

    test('valid image suggests missing sub-fields within existing objects', () => {
      const s = imageSuggestions(VALID_IMAGE);
      const nestedSuggestions = s.filter((sg) => sg.field.includes('.'));
      expect(nestedSuggestions.length).toBeGreaterThan(0);
      expect(nestedSuggestions.some((sg) => sg.field.startsWith('resources.'))).toBe(true);
    });
  });

  describe('empty and invalid input', () => {
    test('empty string returns no diagnostics', () => {
      const result = imageChecker.check('');
      expect(result.diagnostics).toHaveLength(0);
      expect(result.suggestions).toHaveLength(0);
    });

    test('invalid YAML syntax returns error', () => {
      const result = imageChecker.check('group: [unclosed');
      const errs = result.diagnostics.filter((d) => d.severity === Severity.Error);
      expect(errs.length).toBeGreaterThan(0);
    });

    test('non-mapping YAML returns error', () => {
      const result = imageChecker.check('- item1\n- item2');
      const errs = result.diagnostics.filter((d) => d.severity === Severity.Error);
      expect(errs.some((e) => e.message.includes('mapping'))).toBe(true);
    });
  });

  describe('required field errors', () => {
    test('missing group', () => {
      const text = removeLine(VALID_IMAGE, 'group:');
      const errs = imageErrors(text);
      expect(errs.some((e) => e.message.includes("Missing required field: 'group'"))).toBe(true);
    });

    test('missing name', () => {
      const text = removeLine(VALID_IMAGE, 'name:');
      const errs = imageErrors(text);
      expect(errs.some((e) => e.message.includes("Missing required field: 'name'"))).toBe(true);
    });
  });

  describe('enum validation', () => {
    test('invalid scaler value', () => {
      const text = replaceLine(VALID_IMAGE, 'scaler:', 'scaler: Docker');
      const errs = imageErrors(text);
      expect(errs.some((e) => e.message.includes("Invalid scaler value: 'Docker'"))).toBe(true);
    });

    test('invalid display_type value', () => {
      const text = replaceLine(VALID_IMAGE, 'display_type:', 'display_type: Binary');
      const errs = imageErrors(text);
      expect(errs.some((e) => e.message.includes("Invalid display_type value: 'Binary'"))).toBe(true);
    });

    test('valid scaler values pass', () => {
      for (const val of ['K8s', 'BareMetal', 'Windows', 'Kvm', 'External']) {
        const text = replaceLine(VALID_IMAGE, 'scaler:', `scaler: ${val}`);
        const errs = imageErrors(text);
        expect(errs.filter((e) => e.message.includes('scaler'))).toHaveLength(0);
      }
    });
  });

  describe('type validation', () => {
    test('timeout must be a number', () => {
      const text = replaceLine(VALID_IMAGE, 'timeout:', 'timeout: fast');
      const errs = imageErrors(text);
      expect(errs.some((e) => e.message.includes("'timeout' must be a number"))).toBe(true);
    });

    test('collect_logs must be a boolean', () => {
      const text = replaceLine(VALID_IMAGE, 'collect_logs:', 'collect_logs: yes_please');
      const errs = imageErrors(text);
      expect(errs.some((e) => e.message.includes("'collect_logs' must be a boolean"))).toBe(true);
    });
  });

  describe('unknown field warnings', () => {
    test('unknown top-level field', () => {
      const text = VALID_IMAGE + '\nfoobar: baz';
      const warns = imageWarnings(text);
      expect(warns.some((w) => w.message.includes("Unknown image field: 'foobar'"))).toBe(true);
    });

    test('unknown resources sub-field', () => {
      const text = VALID_IMAGE.replace('cpu: 1000', 'cpu: 1000\n    gpus: 4');
      const warns = imageWarnings(text);
      expect(warns.some((w) => w.message.includes("Unknown resources field: 'gpus'"))).toBe(true);
    });
  });

  describe('nested validation', () => {
    test('invalid dependency strategy', () => {
      const text = replaceLine(VALID_IMAGE, 'strategy: Paths', '        strategy: Invalid');
      const errs = imageErrors(text);
      expect(errs.some((e) => e.message.includes("Invalid strategy value: 'Invalid'"))).toBe(true);
    });

    test('invalid lifetime counter', () => {
      const text = replaceLine(VALID_IMAGE, 'counter: jobs', '    counter: infinite');
      const errs = imageErrors(text);
      expect(errs.some((e) => e.message.includes("Invalid counter value: 'infinite'"))).toBe(true);
    });
  });

  describe('suggestions', () => {
    test('missing optional fields are suggested', () => {
      const minimal = `group: test\nname: minimal`;
      const s = imageSuggestions(minimal);
      const fields = s.map((sg) => sg.field);
      expect(fields).toContain('description');
      expect(fields).toContain('image');
      expect(fields).toContain('timeout');
      expect(fields).toContain('scaler');
      expect(fields).toContain('display_type');
      expect(fields).toContain('resources');
    });

    test('missing required fields are suggested when absent', () => {
      const noGroup = `name: minimal`;
      const s = imageSuggestions(noGroup);
      const groupSugg = s.find((sg) => sg.field === 'group');
      expect(groupSugg).toBeDefined();
      expect(groupSugg!.message).toContain('Required');
    });

    test('all known image fields are suggested when missing', () => {
      const minimal = `group: test\nname: minimal`;
      const s = imageSuggestions(minimal);
      const fields = s.map((sg) => sg.field);
      expect(fields).toContain('version');
      expect(fields).toContain('args');
      expect(fields).toContain('security_context');
      expect(fields).toContain('output_collection');
    });

    test('suggestions are sorted by category then alphabetically within each category', () => {
      const minimal = `group: test\nname: minimal`;
      const s = imageSuggestions(minimal);
      const categories = s.map((sg) => sg.category);
      const uniqueCategories = [...new Set(categories)];
      expect(uniqueCategories[0]).toBe('Image');

      for (const cat of uniqueCategories) {
        const catFields = s.filter((sg) => sg.category === cat).map((sg) => sg.field);
        const sorted = [...catFields].sort((a, b) => a.localeCompare(b));
        expect(catFields).toEqual(sorted);
      }
    });

    test('scaler suggestion includes valid values', () => {
      const minimal = `group: test\nname: minimal`;
      const s = imageSuggestions(minimal);
      const scalerSugg = s.find((sg) => sg.field === 'scaler');
      expect(scalerSugg).toBeDefined();
      expect(scalerSugg!.values).toContain('K8s');
      expect(scalerSugg!.values).toContain('BareMetal');
    });

    test('empty scaler suggests valid values', () => {
      const text = replaceLine(VALID_IMAGE, 'scaler:', 'scaler:');
      const s = imageSuggestions(text);
      const scalerSugg = s.find((sg) => sg.field === 'scaler');
      expect(scalerSugg).toBeDefined();
      expect(scalerSugg!.values).toContain('K8s');
    });
  });
});

describe('suggestion schemas', () => {
  describe('image suggestion schemas', () => {
    test('enum suggestions carry enum schema with values', () => {
      const minimal = `group: test\nname: minimal`;
      const s = imageSuggestions(minimal);
      const scalerSugg = s.find((sg) => sg.field === 'scaler');
      expect(scalerSugg?.schema).toBeDefined();
      expect(scalerSugg!.schema!.type).toBe(FieldValueType.Enum);
      expect(scalerSugg!.schema!.enumValues).toContain('K8s');
    });

    test('string suggestions carry string schema', () => {
      const minimal = `group: test\nname: minimal`;
      const s = imageSuggestions(minimal);
      const descSugg = s.find((sg) => sg.field === 'description');
      expect(descSugg?.schema).toBeDefined();
      expect(descSugg!.schema!.type).toBe(FieldValueType.String);
    });

    test('number suggestions carry number schema', () => {
      const minimal = `group: test\nname: minimal`;
      const s = imageSuggestions(minimal);
      const timeoutSugg = s.find((sg) => sg.field === 'timeout');
      expect(timeoutSugg?.schema).toBeDefined();
      expect(timeoutSugg!.schema!.type).toBe(FieldValueType.Number);
    });

    test('object suggestions carry object schema with fields', () => {
      const minimal = `group: test\nname: minimal`;
      const s = imageSuggestions(minimal);
      const lifetimeSugg = s.find((sg) => sg.field === 'lifetime');
      expect(lifetimeSugg?.schema).toBeDefined();
      expect(lifetimeSugg!.schema!.type).toBe(FieldValueType.Object);
      expect(lifetimeSugg!.schema!.fields).toBeDefined();
      expect(lifetimeSugg!.schema!.fields!['counter']).toBeDefined();
      expect(lifetimeSugg!.schema!.fields!['counter'].type).toBe(FieldValueType.Enum);
      expect(lifetimeSugg!.schema!.fields!['amount']).toBeDefined();
      expect(lifetimeSugg!.schema!.fields!['amount'].type).toBe(FieldValueType.Number);
    });

    test('resources suggestion has object schema with numeric sub-fields', () => {
      const minimal = `group: test\nname: minimal`;
      const s = imageSuggestions(minimal);
      const resourcesSugg = s.find((sg) => sg.field === 'resources');
      expect(resourcesSugg?.schema).toBeDefined();
      expect(resourcesSugg!.schema!.type).toBe(FieldValueType.Object);
      expect(resourcesSugg!.schema!.fields!['cpu'].type).toBe(FieldValueType.Number);
      expect(resourcesSugg!.schema!.fields!['memory'].type).toBe(FieldValueType.Number);
      expect(resourcesSugg!.schema!.fields!['worker_slots'].type).toBe(FieldValueType.Number);
    });

    test('empty enum field carries schema', () => {
      const text = replaceLine(VALID_IMAGE, 'scaler:', 'scaler:');
      const s = imageSuggestions(text);
      const scalerSugg = s.find((sg) => sg.field === 'scaler');
      expect(scalerSugg?.schema).toBeDefined();
      expect(scalerSugg!.schema!.type).toBe(FieldValueType.Enum);
    });

    test('nested enum suggestions carry enum schema', () => {
      const text = replaceLine(VALID_IMAGE, 'strategy: Paths', '        strategy:');
      const s = imageSuggestions(text);
      const strategySugg = s.find((sg) => sg.field.endsWith('.strategy'));
      expect(strategySugg?.schema).toBeDefined();
      expect(strategySugg!.schema!.type).toBe(FieldValueType.Enum);
      expect(strategySugg!.schema!.enumValues).toContain('Paths');
    });
  });
});

describe('nested sub-field suggestions', () => {
  test('resources sub-fields are suggested when resources exists', () => {
    const text = `group: test\nname: minimal\nresources:\n    cpu: 1000`;
    const s = imageSuggestions(text);
    const resSuggestions = s.filter((sg) => sg.field.startsWith('resources.'));
    expect(resSuggestions.length).toBeGreaterThan(0);
    expect(resSuggestions.some((sg) => sg.field === 'resources.memory')).toBe(true);
    expect(resSuggestions.some((sg) => sg.field === 'resources.ephemeral_storage')).toBe(true);
  });

  test('lifetime required sub-fields are marked as required', () => {
    const text = `group: test\nname: minimal\nlifetime:\n    counter: jobs`;
    const s = imageSuggestions(text);
    const amountSugg = s.find((sg) => sg.field === 'lifetime.amount');
    expect(amountSugg).toBeDefined();
    expect(amountSugg!.message).toContain('Required');
  });

  test('args sub-fields are suggested when args exists', () => {
    const text = `group: test\nname: minimal\nargs:\n    entrypoint: /run.sh`;
    const s = imageSuggestions(text);
    const argsSuggestions = s.filter((sg) => sg.field.startsWith('args.'));
    expect(argsSuggestions.some((sg) => sg.field === 'args.command')).toBe(true);
    expect(argsSuggestions.some((sg) => sg.field === 'args.output')).toBe(true);
  });

  test('args.output carries enum schema', () => {
    const text = `group: test\nname: minimal\nargs:\n    entrypoint: /run.sh`;
    const s = imageSuggestions(text);
    const outputSugg = s.find((sg) => sg.field === 'args.output');
    expect(outputSugg?.schema).toBeDefined();
    expect(outputSugg!.schema!.type).toBe(FieldValueType.Enum);
    expect(outputSugg!.schema!.enumValues).toContain('None');
    expect(outputSugg!.schema!.enumValues).toContain('Append');
  });

  test('dependency sub-sections are suggested when dependencies exists', () => {
    const text = `group: test\nname: minimal\ndependencies:\n    samples:\n        location: /tmp`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'dependencies.repos')).toBe(true);
    expect(s.some((sg) => sg.field === 'dependencies.tags')).toBe(true);
    expect(s.some((sg) => sg.field === 'dependencies.cache')).toBe(true);
  });

  test('dependency sub-section fields are suggested', () => {
    const text = `group: test\nname: minimal\ndependencies:\n    samples:\n        location: /tmp`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'dependencies.samples.strategy')).toBe(true);
    expect(s.some((sg) => sg.field === 'dependencies.samples.naming')).toBe(true);
    const strategySugg = s.find((sg) => sg.field === 'dependencies.samples.strategy');
    expect(strategySugg!.schema!.type).toBe(FieldValueType.Enum);
    expect(strategySugg!.values).toContain('Paths');
  });

  test('output_collection sub-fields are suggested', () => {
    const text = `group: test\nname: minimal\noutput_collection:\n    handler: Files`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'output_collection.files')).toBe(true);
    expect(s.some((sg) => sg.field === 'output_collection.as_filesystem')).toBe(true);
  });

  test('clean_up required sub-fields are marked as required', () => {
    const text = `group: test\nname: minimal\nclean_up:\n    results: /tmp`;
    const s = imageSuggestions(text);
    const scriptSugg = s.find((sg) => sg.field === 'clean_up.script');
    expect(scriptSugg).toBeDefined();
    expect(scriptSugg!.message).toContain('Required');
  });

  test('kvm required sub-fields are marked as required', () => {
    const text = `group: test\nname: minimal\nscaler: Kvm\nkvm:\n    xml: /path/vm.xml`;
    const s = imageSuggestions(text);
    const qcow2Sugg = s.find((sg) => sg.field === 'kvm.qcow2');
    expect(qcow2Sugg).toBeDefined();
    expect(qcow2Sugg!.message).toContain('Required');
  });

  test('security_context sub-fields are suggested', () => {
    const text = `group: test\nname: minimal\nsecurity_context:\n    user: 1000`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'security_context.group')).toBe(true);
    expect(s.some((sg) => sg.field === 'security_context.allow_privilege_escalation')).toBe(true);
  });
});

describe('null object field replacement suggestions', () => {
  test('null dependencies generates replace suggestion', () => {
    const text = `group: test\nname: minimal\ndependencies:`;
    const s = imageSuggestions(text);
    const depSugg = s.find((sg) => sg.field === 'dependencies' && sg.isReplace);
    expect(depSugg).toBeDefined();
    expect(depSugg!.message).toContain('Populate');
    expect(depSugg!.schema).toBeDefined();
    expect(depSugg!.schema!.type).toBe(FieldValueType.Object);
  });

  test('null resources generates replace suggestion', () => {
    const text = `group: test\nname: minimal\nresources:`;
    const s = imageSuggestions(text);
    const resSugg = s.find((sg) => sg.field === 'resources' && sg.isReplace);
    expect(resSugg).toBeDefined();
    expect(resSugg!.schema!.type).toBe(FieldValueType.Object);
    expect(resSugg!.schema!.fields).toBeDefined();
  });

  test('null security_context generates replace suggestion', () => {
    const text = `group: test\nname: minimal\nsecurity_context:`;
    const s = imageSuggestions(text);
    const scSugg = s.find((sg) => sg.field === 'security_context' && sg.isReplace);
    expect(scSugg).toBeDefined();
  });

  test('null kvm generates replace suggestion', () => {
    const text = `group: test\nname: minimal\nscaler: Kvm\nkvm:`;
    const s = imageSuggestions(text);
    const kvmSugg = s.find((sg) => sg.field === 'kvm' && sg.isReplace);
    expect(kvmSugg).toBeDefined();
  });

  test('null lifetime generates replace suggestion', () => {
    const text = `group: test\nname: minimal\nlifetime:`;
    const s = imageSuggestions(text);
    const ltSugg = s.find((sg) => sg.field === 'lifetime' && sg.isReplace);
    expect(ltSugg).toBeDefined();
  });

  test('null args generates replace suggestion', () => {
    const text = `group: test\nname: minimal\nargs:`;
    const s = imageSuggestions(text);
    const argsSugg = s.find((sg) => sg.field === 'args' && sg.isReplace);
    expect(argsSugg).toBeDefined();
  });

  test('null output_collection generates replace suggestion', () => {
    const text = `group: test\nname: minimal\noutput_collection:`;
    const s = imageSuggestions(text);
    const ocSugg = s.find((sg) => sg.field === 'output_collection' && sg.isReplace);
    expect(ocSugg).toBeDefined();
  });

  test('null child_filters generates replace suggestion', () => {
    const text = `group: test\nname: minimal\nchild_filters:`;
    const s = imageSuggestions(text);
    const cfSugg = s.find((sg) => sg.field === 'child_filters' && sg.isReplace);
    expect(cfSugg).toBeDefined();
  });

  test('null env generates replace suggestion', () => {
    const text = `group: test\nname: minimal\nenv:`;
    const s = imageSuggestions(text);
    const envSugg = s.find((sg) => sg.field === 'env' && sg.isReplace);
    expect(envSugg).toBeDefined();
  });

  test('non-null object does NOT get replace suggestion', () => {
    const text = `group: test\nname: minimal\nresources:\n    cpu: 1000`;
    const s = imageSuggestions(text);
    const resSugg = s.find((sg) => sg.field === 'resources' && sg.isReplace);
    expect(resSugg).toBeUndefined();
  });

  test('explicit null value generates replace suggestion', () => {
    const text = `group: test\nname: minimal\nclean_up: null`;
    const s = imageSuggestions(text);
    const cuSugg = s.find((sg) => sg.field === 'clean_up' && sg.isReplace);
    expect(cuSugg).toBeDefined();
  });
});

describe('env map entry suggestions', () => {
  test('env object offers map entry suggestion', () => {
    const text = `group: test\nname: minimal\nenv:\n    MY_VAR: value`;
    const s = imageSuggestions(text);
    const envSugg = s.find((sg) => sg.field === 'env.VAR_NAME' && sg.isMapEntry);
    expect(envSugg).toBeDefined();
  });
});

describe('auto_tag map entry suggestions', () => {
  test('output_collection with auto_tag offers add auto_tag rule', () => {
    const text = `group: test\nname: minimal\noutput_collection:\n    handler: Files\n    auto_tag:\n        existing_tag:\n            logic: Exists`;
    const s = imageSuggestions(text);
    const atSugg = s.find((sg) => sg.field === 'output_collection.auto_tag.tag-name' && sg.isMapEntry);
    expect(atSugg).toBeDefined();
  });
});

describe('conditional suggestions', () => {
  test('scaler Kvm suggests kvm field when missing', () => {
    const text = `group: test\nname: minimal\nscaler: Kvm`;
    const s = imageSuggestions(text);
    const kvmSugg = s.find((sg) => sg.field === 'kvm' && sg.message.includes('Required'));
    expect(kvmSugg).toBeDefined();
  });

  test('scaler Kvm does not suggest kvm when already present', () => {
    const text = `group: test\nname: minimal\nscaler: Kvm\nkvm:\n    xml: /vm.xml\n    qcow2: /disk.qcow2`;
    const s = imageSuggestions(text);
    const kvmSugg = s.find((sg) => sg.field === 'kvm' && sg.message.includes('Required'));
    expect(kvmSugg).toBeUndefined();
  });

  test('generator true suggests child_filters when missing', () => {
    const text = `group: test\nname: minimal\ngenerator: true`;
    const s = imageSuggestions(text);
    const cfSugg = s.find((sg) => sg.field === 'child_filters' && sg.message.includes('generator'));
    expect(cfSugg).toBeDefined();
  });

  test('output_collection with Files handler suggests files when missing', () => {
    const text = `group: test\nname: minimal\noutput_collection:\n    handler: Files`;
    const s = imageSuggestions(text);
    const filesSugg = s.find((sg) => sg.field === 'output_collection.files' && sg.message.includes('Files'));
    expect(filesSugg).toBeDefined();
  });

  test('children enabled suggests images when missing', () => {
    const text = `group: test\nname: minimal\ndependencies:\n    children:\n        enabled: true\n        location: /tmp`;
    const s = imageSuggestions(text);
    const imgSugg = s.find((sg) => sg.field === 'dependencies.children.images' && sg.message.includes('enabled'));
    expect(imgSugg).toBeDefined();
  });
});

describe('transformVersion', () => {
  test('parses basic semver', () => {
    const result = transformVersion('1.2.3');
    expect(result.valid).toBe(true);
    expect(result.yaml).toContain('SemVer:');
    expect(result.yaml).toContain('major: 1');
    expect(result.yaml).toContain('minor: 2');
    expect(result.yaml).toContain('patch: 3');
    expect(result.json).toContain('"SemVer"');
  });

  test('parses semver with pre-release', () => {
    const result = transformVersion('1.0.0-alpha.1');
    expect(result.valid).toBe(true);
    expect(result.yaml).toContain("pre: 'alpha.1'");
    expect(result.json).toContain('"pre": "alpha.1"');
  });

  test('parses semver with build metadata', () => {
    const result = transformVersion('2.0.0+build.5');
    expect(result.valid).toBe(true);
    expect(result.yaml).toContain("build: 'build.5'");
  });

  test('parses semver with pre-release and build', () => {
    const result = transformVersion('1.0.0-beta.2+build.123');
    expect(result.valid).toBe(true);
    expect(result.yaml).toContain("pre: 'beta.2'");
    expect(result.yaml).toContain("build: 'build.123'");
  });

  test('falls back to Custom for non-semver', () => {
    const result = transformVersion('my-custom-tag');
    expect(result.valid).toBe(true);
    expect(result.yaml).toContain("Custom: 'my-custom-tag'");
    expect(result.json).toContain('"Custom": "my-custom-tag"');
  });

  test('empty string is invalid', () => {
    const result = transformVersion('');
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
  });

  test('version suggestion carries transform', () => {
    const minimal = `group: test\nname: minimal`;
    const s = imageSuggestions(minimal);
    const versionSugg = s.find((sg) => sg.field === 'version');
    expect(versionSugg?.schema?.transform).toBeDefined();
  });
});

describe('version validation', () => {
  test('version as object with SemVer key is valid', () => {
    const text = `group: test\nname: minimal\nversion:\n    SemVer:\n        major: 1\n        minor: 0\n        patch: 0`;
    const errs = imageErrors(text);
    expect(errs.filter((e) => e.message.includes('version'))).toHaveLength(0);
  });

  test('version as object with Custom key is valid', () => {
    const text = `group: test\nname: minimal\nversion:\n    Custom: my-tag`;
    const errs = imageErrors(text);
    expect(errs.filter((e) => e.message.includes('version'))).toHaveLength(0);
  });

  test('version as plain string is valid', () => {
    const text = `group: test\nname: minimal\nversion: '1.0.0'`;
    const errs = imageErrors(text);
    expect(errs.filter((e) => e.message.includes('version'))).toHaveLength(0);
  });

  test('version object without SemVer or Custom warns', () => {
    const text = `group: test\nname: minimal\nversion:\n    invalid: data`;
    const warns = imageWarnings(text);
    expect(warns.some((w) => w.message.includes("'SemVer' or 'Custom'"))).toBe(true);
  });
});

describe('nullable field handling', () => {
  test('null on required string fields produces error', () => {
    const text = `group:\nname: minimal`;
    const errs = imageErrors(text);
    expect(errs.some((e) => e.message.includes("'group' must be a string"))).toBe(true);
  });

  test('null on nullable string fields produces no error', () => {
    const text = `group: test\nname: minimal\nimage:\ndescription:\nmodifiers:`;
    const errs = imageErrors(text);
    expect(errs.some((e) => e.message.includes("'image'"))).toBe(false);
    expect(errs.some((e) => e.message.includes("'description'"))).toBe(false);
    expect(errs.some((e) => e.message.includes("'modifiers'"))).toBe(false);
  });

  test('null on nullable number field produces no error', () => {
    const text = `group: test\nname: minimal\ntimeout:`;
    const errs = imageErrors(text);
    expect(errs.some((e) => e.message.includes("'timeout'"))).toBe(false);
  });

  test('null on non-nullable enum field produces error', () => {
    const text = `group: test\nname: minimal\nscaler:`;
    const errs = imageErrors(text);
    expect(errs.some((e) => e.message.includes("'scaler' must be one of"))).toBe(true);
  });

  test('null on non-nullable boolean field produces error', () => {
    const text = `group: test\nname: minimal\ncollect_logs:`;
    const errs = imageErrors(text);
    expect(errs.some((e) => e.message.includes("'collect_logs' must be a boolean"))).toBe(true);
  });

  test('null security_context.user is allowed', () => {
    const text = `group: test\nname: minimal\nsecurity_context:\n    user:\n    group: 1000`;
    const errs = imageErrors(text);
    expect(errs.some((e) => e.message.includes("'user'"))).toBe(false);
  });

  test('null dependency kwarg is allowed', () => {
    const text = `group: test\nname: minimal\ndependencies:\n    samples:\n        kwarg:\n        strategy: Paths`;
    const errs = imageErrors(text);
    expect(errs.some((e) => e.message.includes("'kwarg'"))).toBe(false);
  });
});

describe('scaler-aware diagnostics', () => {
  test('security_context silently skipped when scaler is External', () => {
    const text = `group: test\nname: minimal\nscaler: External\nsecurity_context:\n    user: 1000`;
    const warns = imageWarnings(text);
    const errs = imageErrors(text);
    expect(warns.some((w) => w.message.includes("'security_context'"))).toBe(false);
    expect(errs.some((e) => e.message.includes("'security_context'"))).toBe(false);
  });

  test('network_policies silently skipped when scaler is Kvm', () => {
    const text = `group: test\nname: minimal\nscaler: Kvm\nnetwork_policies:\n    - allow-dns`;
    const warns = imageWarnings(text);
    const errs = imageErrors(text);
    expect(warns.some((w) => w.message.includes("'network_policies'"))).toBe(false);
    expect(errs.some((e) => e.message.includes("'network_policies'"))).toBe(false);
  });

  test('volumes silently skipped when scaler is BareMetal', () => {
    const text = `group: test\nname: minimal\nscaler: BareMetal\nvolumes:\n    - name: data\n      mount_path: /mnt`;
    const warns = imageWarnings(text);
    const errs = imageErrors(text);
    expect(warns.some((w) => w.message.includes("'volumes'"))).toBe(false);
    expect(errs.some((e) => e.message.includes("'volumes'"))).toBe(false);
  });

  test('kvm silently skipped when scaler is K8s', () => {
    const text = `group: test\nname: minimal\nscaler: K8s\nkvm:\n    xml: /vm.xml\n    qcow2: /disk.qcow2`;
    const warns = imageWarnings(text);
    const errs = imageErrors(text);
    expect(warns.some((w) => w.message.includes("'kvm'"))).toBe(false);
    expect(errs.some((e) => e.message.includes("'kvm'"))).toBe(false);
  });

  test('kvm silently skipped when scaler is not set', () => {
    const text = `group: test\nname: minimal\nkvm:\n    xml: /vm.xml\n    qcow2: /disk.qcow2`;
    const warns = imageWarnings(text);
    const errs = imageErrors(text);
    expect(warns.some((w) => w.message.includes("'kvm'"))).toBe(false);
    expect(errs.some((e) => e.message.includes("'kvm'"))).toBe(false);
  });

  test('security_context validated with K8s scaler', () => {
    const text = `group: test\nname: minimal\nscaler: K8s\nsecurity_context:\n    user: 1000`;
    const warns = imageWarnings(text);
    expect(warns.some((w) => w.message.includes("'security_context'"))).toBe(false);
  });

  test('kvm validated with Kvm scaler', () => {
    const text = `group: test\nname: minimal\nscaler: Kvm\nkvm:\n    xml: /vm.xml\n    qcow2: /disk.qcow2`;
    const warns = imageWarnings(text);
    expect(warns.some((w) => w.message.includes("'kvm'"))).toBe(false);
  });
});

describe('scaler-aware suggestions', () => {
  test('volumes not suggested when scaler is External', () => {
    const text = `group: test\nname: minimal\nscaler: External`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'volumes')).toBe(false);
  });

  test('security_context not suggested when scaler is Kvm', () => {
    const text = `group: test\nname: minimal\nscaler: Kvm`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'security_context')).toBe(false);
  });

  test('kvm not suggested when scaler is K8s', () => {
    const text = `group: test\nname: minimal\nscaler: K8s`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'kvm')).toBe(false);
  });

  test('volumes suggested when scaler is K8s', () => {
    const text = `group: test\nname: minimal\nscaler: K8s`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'volumes')).toBe(true);
  });

  test('volumes suggestion is a structured object list entry (not a plain string array)', () => {
    const text = `group: test\nname: minimal\nscaler: K8s`;
    const vol = imageSuggestions(text).find((sg) => sg.field === 'volumes');
    expect(vol).toBeDefined();
    expect(vol!.isList).toBe(true);
    expect(vol!.schema?.type).toBe(FieldValueType.Object);
    // exposes structured Volume fields incl. the archetype discriminator (rendered as a dropdown)
    expect(vol!.schema?.fields?.archetype?.type).toBe(FieldValueType.Enum);
    expect(vol!.schema?.fields?.mount_path).toBeDefined();
    expect(vol!.schema?.variantField?.field).toBe('archetype');
  });

  test('kvm suggested when scaler is Kvm', () => {
    const text = `group: test\nname: minimal\nscaler: Kvm`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'kvm' && sg.message.includes('Required'))).toBe(true);
  });

  test('network_policies not suggested when scaler is Windows', () => {
    const text = `group: test\nname: minimal\nscaler: Windows`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'network_policies')).toBe(false);
  });

  test('burstable sub-fields not suggested when scaler is BareMetal', () => {
    const text = `group: test\nname: minimal\nscaler: BareMetal\nresources:\n    cpu: 1000`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'resources.burstable')).toBe(false);
    expect(s.some((sg) => sg.field === 'resources.nvidia_gpu')).toBe(false);
  });

  test('burstable sub-fields suggested when scaler is K8s', () => {
    const text = `group: test\nname: minimal\nscaler: K8s\nresources:\n    cpu: 1000`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'resources.burstable')).toBe(true);
    expect(s.some((sg) => sg.field === 'resources.nvidia_gpu')).toBe(true);
  });

  test('security_context gets remove suggestion when scaler is Kvm', () => {
    const text = `group: test\nname: minimal\nscaler: Kvm\nsecurity_context:\n    user: 1000`;
    const s = imageSuggestions(text);
    const removal = s.find((sg) => sg.field === 'security_context' && sg.isRemoval);
    expect(removal).toBeDefined();
    expect(removal!.message).toContain('K8s');
  });

  test('kvm gets remove suggestion when scaler is K8s', () => {
    const text = `group: test\nname: minimal\nscaler: K8s\nkvm:\n    xml: /vm.xml\n    qcow2: /disk.qcow2`;
    const s = imageSuggestions(text);
    const removal = s.find((sg) => sg.field === 'kvm' && sg.isRemoval);
    expect(removal).toBeDefined();
    expect(removal!.message).toContain('Kvm');
  });

  test('volumes gets remove suggestion when scaler is External', () => {
    const text = `group: test\nname: minimal\nscaler: External\nvolumes:\n    - name: data\n      mount_path: /mnt`;
    const s = imageSuggestions(text);
    const removal = s.find((sg) => sg.field === 'volumes' && sg.isRemoval);
    expect(removal).toBeDefined();
    expect(removal!.message).toContain('K8s');
  });

  test('no remove suggestion for security_context when scaler is K8s', () => {
    const text = `group: test\nname: minimal\nscaler: K8s\nsecurity_context:\n    user: 1000`;
    const s = imageSuggestions(text);
    expect(s.some((sg) => sg.field === 'security_context' && sg.isRemoval)).toBe(false);
  });
});

describe('schema/structure fixes (A2, A4, A6, dedupe)', () => {
  test('A2: dependency list sub-fields are StringArray, not string', () => {
    const text = `group: g\nname: n\ndependencies:\n    results:\n        location: /tmp`;
    const s = imageSuggestions(text);
    expect(s.find((x) => x.field === 'dependencies.results.images')?.schema?.type).toBe(FieldValueType.StringArray);
    expect(s.find((x) => x.field === 'dependencies.results.names')?.schema?.type).toBe(FieldValueType.StringArray);
  });

  test('A4: results.kwarg carries the KwargDependency variant schema', () => {
    const text = `group: g\nname: n\ndependencies:\n    results:\n        location: /tmp`;
    const kwarg = imageSuggestions(text).find((x) => x.field === 'dependencies.results.kwarg');
    expect(kwarg?.schema?.variants).toBeDefined();
    expect(Object.keys(kwarg!.schema!.variants!)).toEqual(['None', 'List', 'Map']);
  });

  test('A6: auto_tag logic carries the AutoTagLogic variant schema', () => {
    const text = `group: g\nname: n\noutput_collection:\n    handler: Files\n    auto_tag:\n        t:\n            key: k`;
    const logic = imageSuggestions(text).find((x) => x.field === 'output_collection.auto_tag.t.logic');
    expect(logic?.schema?.variants).toBeDefined();
    expect(logic!.schema!.variants!.Exists).toBeNull();
  });

  test('C: a field is never suggested twice with the same kind', () => {
    for (const text of [
      `group: g\nname: n\nscaler: Kvm`,
      `group: g\nname: n\ngenerator: true`,
      `group: g\nname: n\noutput_collection:\n    handler: Files`,
      `group: g\nname: n\ndependencies:\n    children:\n        enabled: true`,
    ]) {
      const s = imageSuggestions(text);
      const keys = s.map((x) => `${x.field}|${x.isRemoval ? 'r' : ''}${x.isMapEntry ? 'm' : ''}${x.isReplace ? 'p' : ''}`);
      expect(new Set(keys).size).toBe(keys.length);
    }
  });
});

describe('data-carrying enum validation (A4/A6)', () => {
  test('results.kwarg accepts None / {List} / {Map}, rejects a bogus bare string', () => {
    const kwargErrs = (k: string) =>
      imageErrors(`group: g\nname: n\ndependencies:\n  results:\n${k}`).filter((e) => e.message.includes('kwarg'));
    expect(kwargErrs('    kwarg: None')).toHaveLength(0);
    expect(kwargErrs("    kwarg:\n      List: '--r'")).toHaveLength(0);
    expect(kwargErrs('    kwarg:\n      Map:\n        img: x')).toHaveLength(0);
    expect(kwargErrs('    kwarg: bogus').length).toBeGreaterThan(0);
  });

  test('other dependency kwargs remain plain strings', () => {
    const errs = imageErrors(`group: g\nname: n\ndependencies:\n  samples:\n    kwarg: samples`).filter((e) => e.message.includes('kwarg'));
    expect(errs).toHaveLength(0);
  });

  test('auto_tag.logic accepts Exists / {Equal} / {In}, rejects a bogus bare string', () => {
    const logicErrs = (l: string) =>
      imageErrors(`group: g\nname: n\noutput_collection:\n  handler: Files\n  auto_tag:\n    t:\n${l}`).filter((e) =>
        e.message.includes('logic'),
      );
    expect(logicErrs('      logic: Exists')).toHaveLength(0);
    expect(logicErrs('      logic:\n        Equal: 5')).toHaveLength(0);
    expect(logicErrs('      logic:\n        In:\n          - a')).toHaveLength(0);
    expect(logicErrs('      logic: bogus').length).toBeGreaterThan(0);
  });
});
