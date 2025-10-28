from ..models import Assignment


def create_assignment(session, assignment_data):
    assignment = Assignment(**assignment_data)
    session.add(assignment)
    session.commit()
    session.refresh(assignment)
    return assignment
