import { Form } from 'react-bootstrap';
import styles from './TokenViewStyles.module.scss';
import { ScopedTokenRole, ScopedTokenRoleKey, ThoriumDeveloperRoleValue } from '@models/users';
import { useMemo } from 'react';

const defaultDeveloperPermissions: ThoriumDeveloperRoleValue = {
  k8s: false,
  bare_metal: false,
  windows: false,
  external: false,
  kvm: false,
};

type DeveloperPermissionKey = keyof ThoriumDeveloperRoleValue;

type ScopedTokenFormProps = {
  name?: string;
  onNameChange?: (v: string) => void;
  availableGroups: string[];
  selectedGroups: string[];
  onGroupToggle: (group: string) => void;
  role: ScopedTokenRole;
  onRoleChange: (role: ScopedTokenRole) => void;
  expires: string;
  onExpiresChange: (v: string) => void;
  showClearExpires?: boolean;
  clearExpires?: boolean;
  onClearExpiresChange?: (v: boolean) => void;
};

export const ScopedTokenForm = ({
  name,
  onNameChange,
  availableGroups,
  selectedGroups,
  onGroupToggle,
  role,
  onRoleChange,
  expires,
  onExpiresChange,
  showClearExpires,
  clearExpires,
  onClearExpiresChange,
}: ScopedTokenFormProps) => {
  const sortedGroups = useMemo(() => [...availableGroups].sort(), [availableGroups]);
  const roleKey = role === ScopedTokenRoleKey.User ? ScopedTokenRoleKey.User : ScopedTokenRoleKey.Developer;

  const developerPermissions = [
    { key: 'k8s', label: 'K8s' },
    { key: 'bare_metal', label: 'Bare Metal' },
    { key: 'windows', label: 'Windows' },
    { key: 'external', label: 'External' },
    { key: 'kvm', label: 'KVM' },
  ] as const satisfies readonly {
    key: DeveloperPermissionKey;
    label: string;
  }[];

  const onChangeRoleKey = (roleKey: ScopedTokenRoleKey) => {
    switch (roleKey) {
      case ScopedTokenRoleKey.User:
        onRoleChange(ScopedTokenRoleKey.User);
        break;
      case ScopedTokenRoleKey.Developer:
        onRoleChange({ Developer: { ...defaultDeveloperPermissions } });
        break;
      default:
        console.error('Error: attempted to change role key to unsupported role');
    }
  };

  const updateDeveloperPermission = (key: DeveloperPermissionKey, value: boolean) => {
    if (role === ScopedTokenRoleKey.User) return;
    onRoleChange({
      Developer: {
        ...role.Developer,
        [key]: value,
      },
    });
  };

  return (
    <Form>
      {onNameChange !== undefined && (
        <Form.Group className="mb-3">
          <Form.Label>Name</Form.Label>
          <Form.Control type="text" placeholder="my-token" value={name ?? ''} onChange={(e) => onNameChange(e.target.value)} />
          <Form.Text className="text-muted">Scoped token names can contain lower case letters, numbers, and dashes</Form.Text>
        </Form.Group>
      )}
      <Form.Group className="mb-3">
        <Form.Label>Groups</Form.Label>
        {availableGroups.length === 0 ? (
          <p className="text-muted">You are not a member of any groups.</p>
        ) : (
          sortedGroups.map((group) => (
            <Form.Check
              className={styles.group_check}
              key={group}
              type="checkbox"
              label={group}
              checked={selectedGroups.includes(group)}
              onChange={() => onGroupToggle(group)}
            />
          ))
        )}
      </Form.Group>
      <Form.Group className="mb-3">
        <Form.Label>Role</Form.Label>
        <Form.Select value={roleKey} onChange={(e) => onChangeRoleKey(e.target.value as ScopedTokenRoleKey)}>
          {Object.values(ScopedTokenRoleKey).map((selectedRole) => (
            <option key={selectedRole} value={selectedRole}>
              {selectedRole}
            </option>
          ))}
        </Form.Select>
        {roleKey === ScopedTokenRoleKey.Developer && (
          <div className="mt-3">
            {developerPermissions.map(({ key, label }) => (
              <Form.Check
                key={key}
                className={styles.group_check}
                type="checkbox"
                label={label}
                checked={role !== ScopedTokenRoleKey.User && role.Developer[key]}
                onChange={() => role !== ScopedTokenRoleKey.User && updateDeveloperPermission(key, !role.Developer[key])}
              />
            ))}
          </div>
        )}
      </Form.Group>
      <Form.Group className="mb-3">
        <Form.Label>Permanent Expiry (optional)</Form.Label>
        <Form.Control type="date" value={expires} disabled={clearExpires} onChange={(e) => onExpiresChange(e.target.value)} />
      </Form.Group>
      {showClearExpires && onClearExpiresChange && (
        <Form.Check
          type="checkbox"
          label="Clear permanent expiry"
          checked={clearExpires ?? false}
          onChange={(e) => {
            onClearExpiresChange(e.target.checked);
            if (e.target.checked) onExpiresChange('');
          }}
        />
      )}
    </Form>
  );
};
