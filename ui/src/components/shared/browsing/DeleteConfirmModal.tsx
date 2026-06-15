import { Modal, Button } from 'react-bootstrap';

// project imports
import AlertBanner from '@components/shared/alerts/AlertBanner';

// spec: ./browsing.spec.md

export interface DeleteConfirmModalProps {
  show: boolean;
  onHide: () => void;
  onConfirm: () => void;
  itemName: string;
  itemType: string;
  error: string;
}

const DeleteConfirmModal = ({ show, onHide, onConfirm, itemName, itemType, error }: DeleteConfirmModalProps) => (
  <Modal show={show} onHide={onHide} backdrop="static" keyboard={false} centered>
    <Modal.Header closeButton>
      <Modal.Title>Confirm deletion?</Modal.Title>
    </Modal.Header>
    <Modal.Body>
      Do you really want to delete the <b>{itemName}</b> {itemType}?{error !== '' && <AlertBanner className="mt-4">{error}</AlertBanner>}
    </Modal.Body>
    <Modal.Footer className="d-flex justify-content-center">
      <Button className="danger-btn" onClick={onConfirm}>
        Confirm
      </Button>
    </Modal.Footer>
  </Modal>
);

export default DeleteConfirmModal;
