import { Fragment, useEffect, useState } from 'react';
import { Button, Card, Col, Form, Pagination, Row } from 'react-bootstrap';
import AlertBanner, { Severity } from '@components/shared/alerts/AlertBanner';

// project imports
import UploadDropzone from '@components/shared/UploadDropzone';
import { getFileDetails } from '@thorpi/files';
import { downloadAttachment, postFileComments } from '@thorpi/comments';
import type { Comment } from '@models/files';

interface CommentsProps {
  sha256: string;
}
const CommentAlertBanner = ({ commentError }: { commentError: string }) => {
  return (
    <Fragment>
      {commentError == 'Success' && (
        <AlertBanner severity={Severity.Success} className="attachment-card">
          Comment has uploaded successfully!
        </AlertBanner>
      )}
      {commentError != '' && commentError != 'Success' && <AlertBanner className="attachment-card">{commentError}</AlertBanner>}
    </Fragment>
  );
};

const CommentList = ({
  comments,
  page,
  limit,
  getAttachment,
}: {
  comments: Array<Comment>;
  page: number;
  limit: number;
  getAttachment: (commentID: string, name: string, fileID: string) => Promise<void>;
}) => {
  return (
    <Fragment>
      {comments &&
        comments.slice(page * limit, page * limit + limit).map((singleCommentobj, i) => (
          <Card key={i} className="single-comment mb-2 panel">
            <Card.Header>
              {singleCommentobj.author} <i>{singleCommentobj.uploaded}</i>
            </Card.Header>
            <Card.Body>
              <Row key={i}>
                <p key={i}>{singleCommentobj.comment}</p>
              </Row>
              {singleCommentobj &&
                singleCommentobj.attachments &&
                Object.keys(singleCommentobj.attachments).map((name, i) => (
                  <Col key={i}>
                    <a
                      href="#comments"
                      className="text"
                      onClick={() => {
                        void getAttachment(singleCommentobj.id, name, singleCommentobj.attachments[name]);
                      }}
                    >
                      {name}
                    </a>
                  </Col>
                ))}
            </Card.Body>
          </Card>
        ))}
    </Fragment>
  );
};

const Comments = ({ sha256 }: CommentsProps) => {
  const [newComment, setNewComment] = useState('');
  const [filesArray, setFilesArray] = useState<File[]>([]);
  const [comments, setComments] = useState<Comment[]>([]);
  const [limit, setLimit] = useState(0);
  const [maxPage, setMaxPage] = useState(100);
  const [page, setPage] = useState(0);
  const [commentError, setCommentError] = useState('');
  const PAGELIMIT = 10;

  const fetchComments = async () => {
    const fileDetails = await getFileDetails(sha256, setCommentError);
    if (fileDetails && fileDetails.comments) {
      setComments(fileDetails.comments);
      setMaxPage(Math.ceil(fileDetails.comments.length / PAGELIMIT));
      setLimit(PAGELIMIT);
    }
  };

  useEffect(() => {
    void fetchComments();
  }, [sha256]);

  // trigger download of a comment attachment via blob URL
  const getAttachment = async (commentID: string, name: string, fileID: string) => {
    const attachRes = await downloadAttachment(sha256, commentID, fileID, setCommentError);
    if (attachRes && attachRes.data && attachRes.headers) {
      const blob = new Blob([attachRes.data], {
        type: attachRes.headers['content-type'] as string,
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', name);
      document.body.appendChild(link);
      link.click();
      link.parentNode?.removeChild(link);
    }
  };

  // post a new comment with optional attachments
  const handlePost = async (commentValue: string, filesArray: File[]) => {
    const form = new FormData();
    if (commentValue.trim() === '') {
      setCommentError('Comment is Empty');
      return;
    } else {
      form.append('comment', commentValue);
      if (filesArray.length > 0) {
        for (const file of filesArray) {
          form.append('files', file);
        }
      }
      if (await postFileComments(sha256, form, setCommentError)) {
        setCommentError('Success');
        void fetchComments();
        const newPageValue = Math.ceil((comments.length + 1) / PAGELIMIT) - 1;
        if (newPageValue != page && newPageValue != -1) {
          setPage(newPageValue);
        }
      }
    }
  };

  return (
    <div id="comments-tab">
      <div className="comments">
        <CommentList page={page} limit={limit} comments={comments} getAttachment={getAttachment} />
      </div>
      {comments.length == 0 && (
        <Fragment>
          <AlertBanner severity={Severity.Info}>
            <h3>No Comments Available</h3>
          </AlertBanner>
        </Fragment>
      )}
      <Row className="mt-4">
        <Col className="d-flex justify-content-center">
          {comments.length != 0 && (
            <Pagination>
              <Pagination.Prev onClick={() => setPage(page - 1)} disabled={page == 0} />
              <Pagination.Next onClick={() => setPage(page + 1)} disabled={page >= maxPage - 1} />
            </Pagination>
          )}
        </Col>
      </Row>
      <Row>
        <center>
          <Form.Control
            className="comment-entry"
            as="textarea"
            placeholder="Add Comment"
            onChange={(e) => setNewComment(e.target.value)}
            value={newComment}
          />
          <Row>
            <Col>
              <Card className="mt-2 panel attachment-card">
                <Card.Body className="d-flex justify-content-center">
                  <UploadDropzone width={'100%'} onChange={setFilesArray} selectedFiles={[]} />
                </Card.Body>
              </Card>
            </Col>
          </Row>
          <Row className="d-flex justify-content-center mt-2">
            <Col>
              <center>
                <CommentAlertBanner commentError={commentError} />
              </center>
              <Button
                className="mt-3 primary-btn auto-width"
                onClick={() => {
                  void handlePost(newComment, filesArray);
                }}
                disabled={!newComment}
              >
                Post
              </Button>
            </Col>
          </Row>
        </center>
      </Row>
    </div>
  );
};

export default Comments;
