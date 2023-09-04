use crate::{
    circuit::BitsizeCheckCircuit,
    commitment::ChunkedCommitment,
    encryption::{tests::gen_messages,Encryption,Ciphertext},
    keygen::{PreparedDecryptionKey, PreparedEncryptionKey, FkeygenPartial, combineEKs,combineP1share,genP1share,keygenFinal, Fkeygen_dest},
    saver_groth16::{create_proof, verify_proof, get_gs_for_encryption, generate_srs},
    setup::{setup_for_groth16, ChunkedCommitmentGens, EncryptionGens, PreparedEncryptionGens},
    utils::{decompose,chunks_count},
};
use ark_bls12_381::{Bls12_381, G1Affine};
use ark_std::rand::{prelude::StdRng, SeedableRng};
use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
use std::time::Instant;
pub use ark_groth16::{
    prepare_verifying_key, Groth16, PreparedVerifyingKey, Proof, ProvingKey as Groth16ProvingKey,
    VerifyingKey,
};

type Fr = <Bls12_381 as Pairing>::ScalarField;
// test distributed verifiable encryption with addition, re-encryption, and decryption.
#[test]
    fn encrypt_and_ver_distributed_with_addition() {
        fn check(chunk_bit_size: u8) {
            let mut rng = StdRng::seed_from_u64(0u64);
            let gens = EncryptionGens::<Bls12_381>::new_using_rng(&mut rng);
            let n = chunks_count::<Fr>(chunk_bit_size);
            // Get random numbers that are of chunk_bit_size at most
            let msgs = gen_messages(&mut rng, n as usize, chunk_bit_size);
            let msgs_as_field_elems = msgs.iter().map(|m| Fr::from(*m as u64)).collect::<Vec<_>>();
            // another random msg
            let msgs2 = gen_messages(&mut rng, n as usize, chunk_bit_size);
            let msgs_as_field_elems2 = msgs2.iter().map(|m| Fr::from(*m as u64)).collect::<Vec<_>>();
            // basic circuit for testing
            let circuit = BitsizeCheckCircuit::new(chunk_bit_size, Some(n), None, true);
            let snark_srs = generate_srs::<Bls12_381, _, _>(circuit, &gens, &mut rng).unwrap();

            let g_i = get_gs_for_encryption(&snark_srs.pk.vk);
            // party 1 partial key
            let (sk1, ek1) =
                FkeygenPartial(&mut rng, chunk_bit_size, &gens, &g_i, &snark_srs.pk.delta_g1,&snark_srs.gamma_g1).unwrap();
            // party 2 partial key
            let (sk2, ek2) =
                FkeygenPartial(&mut rng, chunk_bit_size, &gens, &g_i, &snark_srs.pk.delta_g1,&snark_srs.gamma_g1).unwrap();
            // combine partial keys (done by both parties)
            let ek3 = combineEKs(&snark_srs.gamma_g1,&[ek1,ek2]);
            // party 1 P1 share
            let p11 = genP1share(&ek3,&sk1);
            // party 2 P1 share
            let p12 = genP1share(&ek3,&sk2);
            // combine P1 (done by both parties)
            let p_combined = combineP1share(&[p11,p12]);
            // generate final ek
            let ek_fin = keygenFinal(&ek3,&p_combined);

            // gen key for destination party (party that can decrypt)
            let (skd, ekd) = Fkeygen_dest(&mut rng, &gens).unwrap();

            // encrypt msg 1
            let (ct, r) =
                Encryption::encrypt_decomposed_message(&mut rng, msgs.clone(), &ek_fin, g_i).unwrap();
            // encrypt msg 2
            let (ct2, r2) =
            Encryption::encrypt_decomposed_message(&mut rng, msgs2.clone(), &ek_fin, g_i).unwrap();
            // combine ciphertexts
            let ct_combined = Encryption::<Bls12_381>::add_cts(&ct,&ct2).unwrap();

            let circuit =
                BitsizeCheckCircuit::new(chunk_bit_size, Some(n), Some(msgs_as_field_elems), true);
            // create proof for msg1
            let start = Instant::now();
            let proof = create_proof(circuit, &r, &snark_srs, &ek_fin, &mut rng).unwrap();
            println!(
                "Time taken to create Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );
            //verify proof for msg1
            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct[0],
                &ct[1..n as usize + 1],
                &ct[n as usize + 1],
                ek_fin.clone(),
                gens.clone(),
            )
            .unwrap();
            let pvk = prepare_verifying_key::<Bls12_381>(&snark_srs.pk.vk);

            let cts = Ciphertext {
                X_r: ct[0],
                enc_chunks: ct[1..n as usize + 1].to_vec(),
                commitment: ct[n as usize + 1],
            };
            verify_proof(&pvk, &proof, &cts).unwrap();
            println!(
                "Time taken to verify Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            let circuit =
                BitsizeCheckCircuit::new(chunk_bit_size, Some(n), Some(msgs_as_field_elems2), true);
            // create proof for msg2
            let start = Instant::now();
            let proof = create_proof(circuit, &r2, &snark_srs, &ek_fin, &mut rng).unwrap();
            println!(
                "Time taken to create Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );
            //verify proof for msg2
            let start = Instant::now();
            Encryption::verify_ciphertext_commitment(
                &ct2[0],
                &ct2[1..n as usize + 1],
                &ct2[n as usize + 1],
                ek_fin.clone(),
                gens.clone(),
            )
            .unwrap();
            let pvk = prepare_verifying_key::<Bls12_381>(&snark_srs.pk.vk);

            let cts = Ciphertext {
                X_r: ct2[0],
                enc_chunks: ct2[1..n as usize + 1].to_vec(),
                commitment: ct2[n as usize + 1],
            };
            verify_proof(&pvk, &proof, &cts).unwrap();
            println!(
                "Time taken to verify Groth16 proof with chunk_bit_size {}: {:?}",
                chunk_bit_size,
                start.elapsed()
            );

            // generate w and g_r for party 1
            let (w1,gr1) = 
            Encryption::re_encrypt_share(
                &mut rng,
                &gens,
                &ct_combined[0],
                &ct_combined[1..msgs.len() + 1],
                &ekd,
                &sk1,
            ).unwrap();
            // generate w and g_r for party 2
            let (w2,gr2) = 
                Encryption::re_encrypt_share(
                    &mut rng,
                    &gens,
                    &ct_combined[0],
                    &ct_combined[1..msgs.len() + 1],
                    &ekd,
                    &sk2,
                ).unwrap();
            // combine w and g_r from both party
            let (wc,grc) = 
                Encryption::<Bls12_381>::combine_re_encrypt_share(
                    &gr1,
                    &gr2,
                    &w1,
                    &w2,
                ).unwrap();
            // perform the re-encryption by combining w and g_r with the ciphertext
            let ct_re_enc = Encryption::combine_ct_with_w(
                &gens,
                &ct_combined[1..msgs.len() + 1],
                &grc,
                &wc,
            ).unwrap();
            // decrypt using destination party's sk
            let m_ = Encryption::<Bls12_381>::decrypt_to_chunks_after_re_encryption(
                &ct_re_enc[0],
                &ct_re_enc[1..n as usize + 1],
                &skd,
                g_i,
                chunk_bit_size*2,
            )
            .unwrap();

            assert_eq!(m_[4], msgs[4]+msgs2[4]);

        }
        check(4);
        // check(8);
        // check(16);
    }