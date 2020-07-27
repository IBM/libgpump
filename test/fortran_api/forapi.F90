! Copyright (C) IBM Corporation 2018. All Rights Reserved
!
!    This program is licensed under the terms of the Eclipse Public License
!    v1.0 as published by the Eclipse Foundation and available at
!    http://www.eclipse.org/legal/epl-v10.html
!
!    
!    
! $COPYRIGHT$
program forapi
use iso_c_binding
use gpump_mod
use mpi
implicit none
!include 'mpif.h'
integer ierror
integer mr_index
integer wx
integer(kind=8) :: buffer_size
integer(kind=8) :: stream
character buffer(30000)
character remote_buffer(30000)
integer(kind=8) :: offset
integer(kind=8) :: remote_offset
integer is_complete

call MPI_Init(ierror)
call gpump_f_init(MPI_COMM_WORLD)
buffer_size = size(buffer)
call gpump_f_register_region(mr_index, LOC(buffer), buffer_size)
call gpump_f_replace_region(1, LOC(buffer), buffer_size)
call gpump_f_connect_propose(0)
call gpump_f_connect_accept(0)
call gpump_f_disconnect(0)
call gpump_f_create_window_propose(0,LOC(buffer), LOC(remote_buffer),buffer_size)
call gpump_f_window_accept(0)
call gpump_f_replace_window_propose(0, LOC(buffer), LOC(remote_buffer), buffer_size)
call gpump_f_window_accept(0)
call gpump_f_create_window_propose_x(0, wx, LOC(buffer), LOC(remote_buffer), buffer_size)
call gpump_f_window_accept_x(0,wx)
call gpump_f_replace_window_propose_x(0, wx, LOC(buffer), LOC(remote_buffer), buffer_size)
call gpump_f_window_accept_x(0,wx)
call gpump_f_cork()
call gpump_f_uncork(stream)
offset = 0
remote_offset = 0
call gpump_f_stream_put(0, stream,offset, remote_offset, buffer_size)
call gpump_f_iput(0, offset, remote_offset, buffer_size)
call gpump_f_stream_wait_put_complete(0, stream)
call gpump_f_cpu_ack_iput(0)
call gpump_f_is_put_complete(0, is_complete)
call gpump_f_wait_put_complete(0)
call gpump_f_stream_get(0, stream,offset, remote_offset, buffer_size)
call gpump_f_iget(0, offset, remote_offset, buffer_size)
call gpump_f_stream_wait_get_complete(0, stream)
call gpump_f_cpu_ack_iget(0)
call gpump_f_is_get_complete(0, is_complete)
call gpump_f_wait_get_complete(0)
call gpump_f_stream_put_x(0,1, stream,offset, remote_offset, buffer_size)
call gpump_f_iput_x(0,1, offset, remote_offset, buffer_size)
call gpump_f_stream_wait_put_complete_x(0,1, stream)
call gpump_f_cpu_ack_iput_x(0,1)
call gpump_f_is_put_complete_x(0,1, is_complete)
call gpump_f_wait_put_complete_x(0,1)
call gpump_f_stream_get_x(0,1, stream,offset, remote_offset, buffer_size)
call gpump_f_iget_x(0,1, offset, remote_offset, buffer_size)
call gpump_f_stream_wait_get_complete_x(0,1, stream)
call gpump_f_cpu_ack_iget_x(0,1)
call gpump_f_is_get_complete_x(0,1, is_complete)
call gpump_f_wait_get_complete_x(0,1)
call gpump_f_stream_send(0, stream, 1, offset, buffer_size)
call gpump_f_isend(0, 1, offset, buffer_size)
call gpump_f_stream_wait_send_complete(0, stream)
call gpump_f_is_send_complete(0, is_complete)
call gpump_f_wait_send_complete(0)
call gpump_f_receive(0, 1, offset, buffer_size)
call gpump_f_stream_wait_recv_complete(0, stream)
call gpump_f_cpu_ack_recv(0)
call gpump_f_is_receive_complete(0, is_complete)
call gpump_f_wait_receive_complete(0)
call gpump_f_destroy_window(0)
call gpump_f_deregister_region(1)
call gpump_f_term()
call gpump_f_init_r(MPI_COMM_WORLD)
buffer_size = size(buffer)
call gpump_f_register_region(mr_index, LOC(buffer), buffer_size)
call gpump_f_replace_region_r(MPI_COMM_WORLD,1, LOC(buffer), buffer_size)
call gpump_f_connect_propose_r(MPI_COMM_WORLD,0)
call gpump_f_connect_accept_r(MPI_COMM_WORLD,0)
call gpump_f_disconnect_r(MPI_COMM_WORLD,0)
call gpump_f_create_window_propose_r(MPI_COMM_WORLD,0,LOC(buffer), LOC(remote_buffer),buffer_size)
call gpump_f_window_accept_r(MPI_COMM_WORLD,0)
call gpump_f_replace_window_propose_r(MPI_COMM_WORLD,0, LOC(buffer), LOC(remote_buffer), buffer_size)
call gpump_f_window_accept_r(MPI_COMM_WORLD,0)
call gpump_f_create_window_propose_rx(MPI_COMM_WORLD,0, wx, LOC(buffer), LOC(remote_buffer), buffer_size)
call gpump_f_window_accept_rx(MPI_COMM_WORLD,0, wx)
call gpump_f_replace_window_propose_rx(MPI_COMM_WORLD,0, 0, LOC(buffer), LOC(remote_buffer), buffer_size)
call gpump_f_window_accept_rx(MPI_COMM_WORLD,0, wx)
call gpump_f_cork()
call gpump_f_uncork(stream)
offset = 0
remote_offset = 0
call gpump_f_stream_put_r(MPI_COMM_WORLD,0, stream,offset, remote_offset, buffer_size)
call gpump_f_iput_r(MPI_COMM_WORLD,0, offset, remote_offset, buffer_size)
call gpump_f_stream_wait_put_complete_r(MPI_COMM_WORLD,0, stream)
call gpump_f_cpu_ack_iput_r(MPI_COMM_WORLD,0)
call gpump_f_is_put_complete_r(MPI_COMM_WORLD,0, is_complete)
call gpump_f_wait_put_complete_r(MPI_COMM_WORLD,0)
call gpump_f_stream_get_r(MPI_COMM_WORLD,0, stream,offset, remote_offset, buffer_size)
call gpump_f_iget_r(MPI_COMM_WORLD,0, offset, remote_offset, buffer_size)
call gpump_f_stream_wait_get_complete_r(MPI_COMM_WORLD,0, stream)
call gpump_f_cpu_ack_iget_r(MPI_COMM_WORLD,0)
call gpump_f_is_get_complete_r(MPI_COMM_WORLD,0, is_complete)
call gpump_f_wait_get_complete_r(MPI_COMM_WORLD,0)
call gpump_f_stream_put_rx(MPI_COMM_WORLD,0,1, stream,offset, remote_offset, buffer_size)
call gpump_f_iput_rx(MPI_COMM_WORLD,0,1, offset, remote_offset, buffer_size)
call gpump_f_stream_wait_put_complete_rx(MPI_COMM_WORLD,0,1, stream)
call gpump_f_cpu_ack_iput_rx(MPI_COMM_WORLD,0,1)
call gpump_f_is_put_complete_rx(MPI_COMM_WORLD,0,1, is_complete)
call gpump_f_wait_put_complete_rx(MPI_COMM_WORLD,0,1)
call gpump_f_stream_get_rx(MPI_COMM_WORLD,0,1, stream,offset, remote_offset, buffer_size)
call gpump_f_iget_rx(MPI_COMM_WORLD,0,1, offset, remote_offset, buffer_size)
call gpump_f_stream_wait_get_complete_rx(MPI_COMM_WORLD,0,1, stream)
call gpump_f_cpu_ack_iget_rx(MPI_COMM_WORLD,0,1)
call gpump_f_is_get_complete_rx(MPI_COMM_WORLD,0,1, is_complete)
call gpump_f_wait_get_complete_rx(MPI_COMM_WORLD,0,1)
call gpump_f_stream_send_r(MPI_COMM_WORLD,0, stream, 1, offset, buffer_size)
call gpump_f_isend_r(MPI_COMM_WORLD,0, 1, offset, buffer_size)
call gpump_f_stream_wait_send_complete_r(MPI_COMM_WORLD,0, stream)
call gpump_f_is_send_complete_r(MPI_COMM_WORLD,0, is_complete)
call gpump_f_wait_send_complete_r(MPI_COMM_WORLD,0)
call gpump_f_receive_r(MPI_COMM_WORLD,0, 1, offset, buffer_size)
call gpump_f_stream_wait_recv_complete_r(MPI_COMM_WORLD,0, stream)
call gpump_f_cpu_ack_recv_r(MPI_COMM_WORLD,0)
call gpump_f_is_receive_complete_r(MPI_COMM_WORLD,0, is_complete)
call gpump_f_wait_receive_complete_r(MPI_COMM_WORLD,0)
call gpump_f_destroy_window_r(MPI_COMM_WORLD,0)
call gpump_f_deregister_region_r(MPI_COMM_WORLD,1)
call gpump_f_term()
call MPI_Finalize(ierror)

end program forapi
