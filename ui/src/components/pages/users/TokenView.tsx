import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Alert, Badge, Button, Col, Container, Modal, OverlayTrigger, Row, Tooltip } from 'react-bootstrap';
import styled from 'styled-components';
import { FaClipboard, FaCheck } from 'react-icons/fa6';

// project imports
import Subtitle from '@components/shared/titles/Subtitle';
import { useAuth } from '@utilities/auth';
import { createScopedToken, deleteScopedToken, listScopedTokens, updateScopedToken } from '@thorpi/users';
import { ScopedToken, ScopedTokenRole, ScopedTokenRoleKey, ScopedTokenUpdate, UserInfo } from '@models/users';
import { ScopedTokenForm } from './ScopedTokenForm';
import styles from './TokenViewStyles.module.scss';
import { FaQuestionCircle } from 'react-icons/fa';
import { toast } from 'react-toastify';

// ─── styled helpers ───────────────────────────────────────────────────────────

const HiddenToken = styled.p`
  color: var(--thorium-secondary-text);
  overflow-wrap: anywhere;
`;

const WrapToken = styled.p`
  overflow-wrap: anywhere;
`;

const ScopedSection = styled.div`
  margin-top: 1.5rem;
`;

// ─── Shared form ──────────────────────────────────────────────────────────────
// ─── Root token ───────────────────────────────────────────────────────────────

const RevokeTokenModal = ({ show, onHide }: { show: boolean; onHide: () => void }) => {
  const { revoke } = useAuth();
  const navigate = useNavigate();
  const handleRevoke = () => {
    void revoke().then(() => {
      void navigate('/');
    });
  };
  return (
    <Modal show={show} onHide={onHide}>
      <Modal.Header closeButton>
        <Modal.Title>Revoke Your Token?</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        Revoking your token will automatically log you out of this page and any currently running or queued analysis jobs (reactions) may
        fail. Are you sure?
      </Modal.Body>
      <Modal.Footer className="d-flex justify-content-center">
        <Button className="danger-btn" onClick={() => handleRevoke()}>
          Confirm
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

const RootToken = () => {
  const [showRevokeModal, setShowRevokeModal] = useState(false);
  const [tokenShowing, setTokenShowing] = useState(false);
  const { userInfo, clearScopedToken, scopedToken } = useAuth();

  return (
    <Container>
      <Row>
        <Col xs={2}>
          <Subtitle>Token</Subtitle>
        </Col>
        <Col xs={10}>
          <Row>
            <Col>
              <div>
                {tokenShowing ? (
                  <WrapToken>{userInfo?.token}</WrapToken>
                ) : (
                  <HiddenToken>****************************************************************</HiddenToken>
                )}
              </div>
            </Col>
          </Row>
        </Col>
      </Row>
      <Row>
        <Col className={styles.button_row}>
          {scopedToken && <Button onClick={() => clearScopedToken()}>Use Root Token</Button>}
          <Button className="primary-btn" onClick={() => setTokenShowing(!tokenShowing)}>
            {tokenShowing ? 'Hide' : 'Show'}
          </Button>
          <Button className="danger-btn" onClick={() => setShowRevokeModal(true)}>
            Revoke
          </Button>
        </Col>
      </Row>
      <Row className="pt-3">
        <Col xs={2}>
          <Subtitle>Expiry</Subtitle>
        </Col>
        <Col>
          <p>{userInfo?.token_expiration}</p>
        </Col>
      </Row>
      <RevokeTokenModal show={showRevokeModal} onHide={() => setShowRevokeModal(false)} />
    </Container>
  );
};

// ─── Create modal ─────────────────────────────────────────────────────────────

type CreateModalProps = {
  show: boolean;
  availableGroups: string[];
  onHide: () => void;
  onCreated: () => void;
};

const CreateTokenModal = ({ show, availableGroups, onHide, onCreated }: CreateModalProps) => {
  const [name, setName] = useState('');
  const [selectedGroups, setSelectedGroups] = useState<string[]>([]);
  const [role, setRole] = useState<ScopedTokenRole>(ScopedTokenRoleKey.User);
  const [expires, setExpires] = useState('');
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const reset = () => {
    setName('');
    setSelectedGroups([]);
    setExpires('');
    setError('');
    setSubmitting(false);
    setRole(ScopedTokenRoleKey.User);
  };

  const handleHide = () => {
    reset();
    onHide();
  };

  const toggleGroup = (group: string) => {
    setSelectedGroups((prev) => (prev.includes(group) ? prev.filter((g) => g !== group) : [...prev, group]));
  };

  const handleSubmit = async () => {
    setError('');
    if (!name.trim()) {
      setError('Name is required.');
      return;
    }
    if (selectedGroups.length === 0) {
      setError('At least one group must be selected.');
      return;
    }
    setSubmitting(true);
    const expiresIso = expires ? `${expires}T00:00:00Z` : undefined;
    const result = await createScopedToken(
      {
        role: role,
        name: name.trim(),
        groups: selectedGroups,
        expires: expiresIso,
      },
      (msg) => setError(msg),
    );
    setSubmitting(false);
    if (result) {
      reset();
      onCreated();
    }
  };

  return (
    <Modal show={show} onHide={handleHide}>
      <Modal.Header closeButton>
        <Modal.Title>New Scoped Token</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {error && <Alert variant="danger">{error}</Alert>}
        <ScopedTokenForm
          name={name}
          onNameChange={setName}
          availableGroups={availableGroups}
          selectedGroups={selectedGroups}
          role={role}
          onRoleChange={setRole}
          onGroupToggle={toggleGroup}
          expires={expires}
          onExpiresChange={setExpires}
        />
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={handleHide}>
          Cancel
        </Button>
        <Button className="primary-btn" onClick={() => void handleSubmit()} disabled={submitting}>
          {submitting ? 'Creating…' : 'Create'}
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

// ─── Delete modal ─────────────────────────────────────────────────────────────

type DeleteModalProps = {
  token: ScopedToken | null;
  onHide: () => void;
  onDeleted: () => void;
};

const DeleteTokenModal = ({ token, onHide, onDeleted }: DeleteModalProps) => {
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleHide = () => {
    setError('');
    setSubmitting(false);
    onHide();
  };

  const handleDelete = async () => {
    if (!token) return;
    setError('');
    setSubmitting(true);
    const ok = await deleteScopedToken(token.name, (msg) => setError(msg));
    setSubmitting(false);
    if (ok) {
      handleHide();
      onDeleted();
    }
  };

  return (
    <Modal show={!!token} onHide={handleHide}>
      <Modal.Header closeButton>
        <Modal.Title>Delete Scoped Token?</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {error && <Alert variant="danger">{error}</Alert>}
        {token && (
          <p>
            Delete <strong>{token.name}</strong>? This cannot be undone.
          </p>
        )}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={handleHide}>
          Cancel
        </Button>
        <Button className="danger-btn" onClick={() => void handleDelete()} disabled={submitting}>
          {submitting ? 'Deleting…' : 'Delete'}
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

// ─── Edit modal ───────────────────────────────────────────────────────────────

type EditModalProps = {
  token: ScopedToken | null;
  availableGroups: string[];
  onHide: () => void;
  onUpdated: () => void;
};

const EditTokenModal = ({ token, availableGroups, onHide, onUpdated }: EditModalProps) => {
  const [selectedGroups, setSelectedGroups] = useState<string[]>([]);
  const [expires, setExpires] = useState('');
  const [role, setRole] = useState<ScopedTokenRole>(token ? token.role : ScopedTokenRoleKey.User);
  const [clearExpires, setClearExpires] = useState(false);
  const [error, setError] = useState('');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    //refresh any time token selected
    if (token) {
      setSelectedGroups(token.groups);
      // token.expires is a full ISO string — take YYYY-MM-DD prefix
      setExpires(token.expires ? token.expires.slice(0, 10) : '');
      setClearExpires(false);
      setError('');
      setSubmitting(false);
      setRole(token.role);
    }
  }, [token]);

  const handleHide = () => {
    setError('');
    setSubmitting(false);
    onHide();
  };

  const toggleGroup = (group: string) => {
    setSelectedGroups((prev) => (prev.includes(group) ? prev.filter((g) => g !== group) : [...prev, group]));
  };

  const handleSubmit = async () => {
    if (!token) return;
    if (selectedGroups.length === 0) {
      setError('At least one group must be selected.');
      return;
    }
    setError('');
    setSubmitting(true);

    const original = new Set(token.groups);
    const updated = new Set(selectedGroups);
    const add_groups = [...updated].filter((g) => !original.has(g));
    const remove_groups = [...original].filter((g) => !updated.has(g));

    const update: ScopedTokenUpdate = { role: role, add_groups, remove_groups, clear_expires: clearExpires };
    if (expires && !clearExpires) {
      update.expires = `${expires}T00:00:00Z`;
    }

    const result = await updateScopedToken(token.name, update, (msg) => setError(msg));
    setSubmitting(false);
    if (result) {
      handleHide();
      onUpdated();
    }
  };

  return (
    <Modal show={!!token} onHide={handleHide}>
      <Modal.Header closeButton>
        <Modal.Title>Edit Scoped Token{token ? ` — ${token.name}` : ''}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        {error && <Alert variant="danger">{error}</Alert>}
        <ScopedTokenForm
          availableGroups={availableGroups}
          selectedGroups={selectedGroups}
          onGroupToggle={toggleGroup}
          role={role}
          onRoleChange={setRole}
          expires={expires}
          onExpiresChange={setExpires}
          showClearExpires={!!token?.expires}
          clearExpires={clearExpires}
          onClearExpiresChange={(v) => setClearExpires(v)}
        />
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={handleHide}>
          Cancel
        </Button>
        <Button className="primary-btn" onClick={() => void handleSubmit()} disabled={submitting}>
          {submitting ? 'Saving…' : 'Save'}
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

type SelectModalProps = {
  token: ScopedToken | null;
  onHide: () => void;
};

const SelectModal = ({ token, onHide }: SelectModalProps) => {
  const { setScopedToken } = useAuth();

  const handleSubmit = () => {
    if (!token) return;
    setScopedToken(token);
    onHide();
  };

  const groups = [...(token?.groups ?? [])].sort().map((g) => {
    return <li key={g}>{g}</li>;
  });

  return (
    <Modal show={!!token} onHide={onHide}>
      <Modal.Header closeButton>
        <Modal.Title>Activate Scoped Token{token ? ` — ${token.name}` : ''}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        Select "{token?.name}" as token? This will limit visibility to the following groups:
        <ul className={styles.auto_overflow}>{groups}</ul>
        Several features of Thorium UI may be unusable with a scoped token. To re-enable full functionality, disable the scoped token.
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onHide}>
          Cancel
        </Button>
        <Button variant="primary" onClick={() => void handleSubmit()}>
          Change Token
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

function formatExpire(expires: string | null): string {
  if (!expires) return '-';
  return expires.length > 10 ? expires.slice(0, 10) : expires;
}

const TokenGroupDisplay = ({ groups }: { groups: string[] }) => {
  if (groups.length === 0) {
    return <span className={styles.muted_value}>-</span>;
  }

  const sortedGroups = [...groups].sort();
  const visibleGroups = sortedGroups.slice(0, 2);
  const hasMore = sortedGroups.length > 2;

  const description = sortedGroups.join('\n');

  const tooltip = <Tooltip className={styles.badge_tip}>{description}</Tooltip>;

  return (
    <div className={styles.group_list}>
      {visibleGroups.map((g) => (
        <Badge key={g} pill bg="" className={`bg-blue ${styles.group_badge}`} title={g}>
          {g}
        </Badge>
      ))}
      {hasMore && (
        <OverlayTrigger overlay={tooltip}>
          <Badge pill bg="" className={`bg-blue ${styles.group_badge}`}>
            {sortedGroups.length - 2} more...
          </Badge>
        </OverlayTrigger>
      )}
    </div>
  );
};

const TokenRow = ({
  tok,
  selected,
  onEdit,
  onDelete,
  onSelect,
}: {
  tok: ScopedToken;
  selected: boolean;
  onEdit: (t: ScopedToken) => void;
  onDelete: (t: ScopedToken) => void;
  onSelect: (t: ScopedToken) => void;
}) => {
  const [copied, setCopied] = useState(false);
  const { clearScopedToken, scopedToken } = useAuth();

  const handleCopy = () => {
    void navigator.clipboard
      .writeText(tok.token)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
        toast.success(`Copied value of token "${tok.name}" to clipboard!`);
      })
      .catch(() => {
        toast.error(`Error copying value of token "${tok.name}" to clipboard.`);
      });
  };

  const handleSelect = () => {
    if (selected) {
      clearScopedToken();
    } else {
      onSelect(tok);
    }
  };

  const selectedName = selected ? 'Deactivate' : scopedToken !== undefined ? 'Switch' : 'Activate';

  return (
    <tr className={selected ? styles.selected_row : ''}>
      <td className={styles.name_cell}>{tok.name}</td>
      <td className={styles.token_cell}>
        <Button size="sm" variant="link" className="p-1" title="Copy token" onClick={handleCopy}>
          {copied ? <FaCheck /> : <FaClipboard />}
        </Button>
        <span className={styles.masked_token}>***********</span>
      </td>
      <td>
        <TokenRoleView role={tok.role} />
      </td>
      <td className={styles.groups_cell}>
        <TokenGroupDisplay groups={tok.groups} />
      </td>
      <td className={styles.expiry_cell}>{formatExpire(tok.token_expiration)}</td>
      <td className={styles.expiry_cell}>{formatExpire(tok.expires)}</td>
      <td className={styles.actions_cell}>
        <div className={styles.action_buttons}>
          <Button size="sm" className="secondary-btn" onClick={handleSelect}>
            {selectedName}
          </Button>
          <Button size="sm" className="secondary-btn" onClick={() => onEdit(tok)} disabled={scopedToken !== undefined}>
            Edit
          </Button>
          <Button size="sm" className="danger-btn" onClick={() => onDelete(tok)} disabled={scopedToken !== undefined}>
            Delete
          </Button>
        </div>
      </td>
    </tr>
  );
};

const TokenRoleView = ({ role }: { role: ScopedTokenRole }) => {
  if (role === ScopedTokenRoleKey.User) {
    return <span>{role}</span>;
  }
  return (
    <>
      <OverlayTrigger
        overlay={
          <Tooltip id="developer-role-tooltip">
            <div className="text-start">
              {Object.entries(role.Developer).map(([key, value]) => (
                <div key={key}>
                  <strong>{key}:</strong> {value ? 'true' : 'false'}
                </div>
              ))}
            </div>
          </Tooltip>
        }
      >
        <span>
          Developer
          <span className={styles.developer_question}>
            <FaQuestionCircle></FaQuestionCircle>
          </span>
        </span>
      </OverlayTrigger>
    </>
  );
};

const getAvailGroups = (userInfo: UserInfo | null): string[] => {
  if (!userInfo) return [];
  if (userInfo.actual_groups !== undefined) return userInfo.actual_groups;
  return userInfo.groups;
};

const sortTokens = (tokens: ScopedToken[], scopedToken?: ScopedToken) =>
  [...tokens].sort((a, b) => {
    if (scopedToken) {
      if (scopedToken.name === a.name) return -1;
      if (scopedToken.name === b.name) return 1;
    }
    return a.name.localeCompare(b.name);
  });

const ScopedTokens = () => {
  const [tokens, setTokens] = useState<ScopedToken[]>([]);
  const [listError, setListError] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<ScopedToken | null>(null);
  const [editTarget, setEditTarget] = useState<ScopedToken | null>(null);
  const [selectTarget, setSelectTarget] = useState<ScopedToken | null>(null);

  const { scopedToken, userInfo } = useAuth();

  const availableGroups = getAvailGroups(userInfo);

  const refresh = useCallback(() => {
    void listScopedTokens((msg) => setListError(msg)).then((result) => {
      if (result) {
        setListError('');
        setTokens(result);
      }
    });
  }, []);

  const sortedTokens = useMemo(() => sortTokens(tokens, scopedToken), [tokens, scopedToken]);

  useEffect(() => {
    refresh();
  }, [refresh, scopedToken]);

  return (
    <ScopedSection className={styles.token_section}>
      <div className={styles.token_section_header}>
        <Subtitle>Scoped Tokens</Subtitle>
        <Button
          variant="secondary"
          size="sm"
          className={styles.create_button}
          onClick={() => setShowCreate(true)}
          disabled={scopedToken !== undefined}
        >
          New Scoped Token
        </Button>
      </div>
      {listError && (
        <Row>
          <Col>
            <Alert variant="danger">{listError}</Alert>
          </Col>
        </Row>
      )}
      {sortedTokens.length === 0 ? (
        <p className="pt-3">No scoped tokens yet.</p>
      ) : (
        <div className={styles.table_wrap}>
          <table className={styles.token_view_table}>
            <thead>
              <tr>
                <th>Name</th>
                <th>Token</th>
                <th>Role</th>
                <th>Groups</th>
                <th>Value Expiry</th>
                <th>Token Expiry</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {sortedTokens.map((tok) => (
                <TokenRow
                  key={tok.name}
                  selected={scopedToken?.name === tok.name}
                  tok={tok}
                  onEdit={setEditTarget}
                  onDelete={setDeleteTarget}
                  onSelect={setSelectTarget}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      <CreateTokenModal
        show={showCreate}
        availableGroups={availableGroups}
        onHide={() => setShowCreate(false)}
        onCreated={() => {
          setShowCreate(false);
          refresh();
        }}
      />
      <DeleteTokenModal
        token={deleteTarget}
        onHide={() => setDeleteTarget(null)}
        onDeleted={() => {
          setDeleteTarget(null);
          refresh();
        }}
      />
      <EditTokenModal
        token={editTarget}
        availableGroups={availableGroups}
        onHide={() => setEditTarget(null)}
        onUpdated={() => {
          setEditTarget(null);
          refresh();
        }}
      />
      <SelectModal token={selectTarget} onHide={() => setSelectTarget(null)} />
    </ScopedSection>
  );
};

// ─── Public export ────────────────────────────────────────────────────────────

const TokenView = () => {
  return (
    <>
      <RootToken />
      <hr />
      <ScopedTokens />
    </>
  );
};

export default TokenView;
